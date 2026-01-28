// isolation-rs
//
// Computes topographic isolation for peaks using a DEM (GeoTIFF/any GDAL-supported raster).
//
// Isolation definition here:
//   isolation = min distance to any DEM cell with elevation >= peak_elev,
//   searched up to max_radius.
//
// Robustness tweak for imprecise peak coordinates:
//   --min-distance M
//   normally, ignore DEM blockers closer than M.
//   BUT if there exists another peak with elevation >= current peak within M,
//   then the effective minimum becomes the distance to the nearest such peak.
//
// Parallelism + cache locality:
// - Peaks are clustered by DEM "super-block" (groups of GDAL blocks).
// - Clusters are processed in parallel; each worker opens its own GDAL Dataset and has its own BlockCache.
// - Output order is NOT preserved (lines are printed as soon as a peak is computed).
//
// Input format (stdin): semicolon-separated
//   <id>;<lon>;<lat>[;<elev>]
//   (input elevation is optional and ignored; DEM elevation is used)
// Lines starting with # are ignored.
//
// Output (stdout): semicolon-separated
//   id;lon;lat;elev;isolation_m
//   (elev is DEM value multiplied by GeoTIFF scale, if present)
//
// Notes:
// - Assumes north-up geotransform (no rotation). If your DEM is rotated, reproject it.

use anyhow::{Context, Result, anyhow, bail};
use clap::Parser;
use gdal::Dataset;
use geo::{Distance, Euclidean, Point};
use proj::Proj;
use rayon::{
    ThreadPoolBuilder,
    iter::{IntoParallelIterator, ParallelIterator},
};
use rstar::{AABB, PointDistance, RTree, RTreeObject};
use std::{
    cell::RefCell,
    collections::{HashMap, VecDeque},
    f64,
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    num::NonZero,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

type DataType = i32;

#[derive(Parser, Debug, Clone)]
#[command(name = "isolation-rs")]
#[command(about = "Compute peak isolations from a DEM", long_about = None)]
struct Args {
    /// Input peaks CSV
    #[arg(long)]
    input: PathBuf,

    /// Output peaks CSV
    #[arg(long)]
    output: PathBuf,

    /// DEM path (GeoTIFF etc.)
    #[arg(long)]
    dem: PathBuf,

    /// Maximum search radius in meters
    #[arg(long, default_value_t = 100_000.0)]
    max_radius: f64,

    /// Minimum distance in meters for accepting DEM blockers (see header comment)
    #[arg(long, default_value_t = 0.0)]
    min_distance: f64,

    /// Maximum number of DEM blocks kept in memory (per thread)
    #[arg(long, default_value_t = 256)]
    block_cache: usize,

    /// Worker threads (default: available parallelism)
    #[arg(long)]
    threads: Option<usize>,

    /// Cluster granularity as a shift on GDAL block coords.
    /// 0 => 1x1 blocks per cluster, 3 => 8x8 blocks per cluster.
    #[arg(long, default_value_t = 3)]
    cluster_shift: u8,
}

#[derive(Clone, Copy, Debug)]
struct Peak {
    id: u64,
    point: Point,
    elev: DataType,
}

#[derive(Clone, Copy, Debug)]
struct PeakTask {
    p: Peak,
    effective_min_m: f64,
    search_radius_m: f64,
}

impl RTreeObject for Peak {
    type Envelope = AABB<Point>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_point(self.point)
    }
}

impl PointDistance for Peak {
    fn distance_2(
        &self,
        point: &<Self::Envelope as rstar::Envelope>::Point,
    ) -> <<Self::Envelope as rstar::Envelope>::Point as rstar::Point>::Scalar {
        let dx = point.x() - self.point.x();
        let dy = point.y() - self.point.y();
        dx * dx + dy * dy
    }
}

struct WorkerState {
    ds: Dataset,
    raster_size: (isize, isize),
    gt: [f64; 6],
    block_size: (isize, isize),
    cache: BlockCache<DataType>,
}

thread_local! {
    static WORKER: RefCell<Option<WorkerState>> = const { RefCell::new(None) };
}

fn main() -> Result<()> {
    let args = Args::parse();

    let proj =
        Proj::new_known_crs("EPSG:4326", "EPSG:3035", None).context("Error creating projection")?;

    // Open once in the main thread to read georeferencing + block layout for clustering.
    let ds = Dataset::open(&args.dem).context("Error opening dataset")?;

    let band = ds.rasterband(1).context("Error getting rasterband 1")?;

    let gt = ds.geo_transform().context("Error getting transformation")?;

    // North-up only: gt[2] and gt[4] are 0.
    if gt[2] != 0.0 || gt[4] != 0.0 {
        bail!("Rotated geotransform not supported (reproject/warp to north-up).");
    }

    let (width, height) = ds.raster_size();

    println!("DEM dimensions: {width}x{height}");

    let (block_w, block_h) = band.block_size();

    println!("Block size: {block_w}x{block_h}");

    let scale = band.scale().unwrap_or(1.0);

    let peaks = read_peaks(
        &args.input,
        &proj,
        &band,
        gt,
        (width as isize, height as isize),
        (block_w as isize, block_h as isize),
        args.block_cache,
    )
    .context("Error reading peaks")?;

    println!("Peaks: {}", peaks.len());

    if peaks.is_empty() {
        return Ok(());
    }

    let mut tree = RTree::new();

    for peak in peaks {
        tree.insert(peak);
    }

    println!("Indexed.");

    let max_radius2 = args.max_radius.powf(2.0);

    let mut cnt: u32 = 0;

    let tasks = tree
        .iter()
        .map(|p| {
            let mut effective_min_m = f64::NAN;

            for (peak, distance2) in tree.nearest_neighbor_iter_with_distance_2(&p.point) {
                if peak.id == p.id {
                    continue;
                }

                if f64::is_nan(effective_min_m) {
                    effective_min_m = distance2.sqrt();
                }

                if peak.elev > p.elev || distance2 > max_radius2 {
                    let search_radius_m =
                        distance2.sqrt().min(args.max_radius).max(effective_min_m);

                    cnt += 1;

                    return PeakTask {
                        p: *p,
                        effective_min_m,
                        search_radius_m,
                    };
                }
            }

            cnt += 1;

            PeakTask {
                p: *p,
                effective_min_m,
                search_radius_m: args.max_radius,
            }
        })
        .collect::<Vec<_>>();

    println!("Tasks: {}", tasks.len());

    // Cluster tasks by super-block for cache locality.
    let clusters = cluster_tasks(
        &tasks,
        gt,
        (width as isize, height as isize),
        (block_w as isize, block_h as isize),
        args.cluster_shift,
    );

    println!("Clusters: {}", clusters.len());

    // --- Parallel execution using rayon ---
    // Each rayon worker keeps its own GDAL Dataset + BlockCache via thread-local storage.

    let threads = args
        .threads
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(NonZero::get)
                .unwrap_or(1)
        })
        .max(1);

    let out = Arc::new(Mutex::new(BufWriter::new(
        File::create(args.output).context("Error creating output file")?,
    )));

    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .context("Error building thread pool")?
        .install(|| {
            clusters.into_par_iter().for_each(|cluster| {
                for t in cluster {
                    let iso = WORKER.with(|cell| {
                        if cell.borrow().is_none() {
                            *cell.borrow_mut() = Some(WorkerState {
                                ds: Dataset::open(&args.dem).expect("open DEM"),
                                raster_size: (width as isize, height as isize),
                                gt,
                                block_size: (block_w as isize, block_h as isize),
                                cache: BlockCache::<DataType>::new(args.block_cache),
                            });
                        }

                        let mut borrow = cell.borrow_mut();

                        let state = borrow.as_mut().unwrap();

                        isolation_by_dem(state, t.p, t.search_radius_m, t.effective_min_m)
                    });

                    match iso {
                        Ok(iso) => {
                            let mut w = out.lock().unwrap();
                            let elev_scaled = (t.p.elev as f64) * scale;
                            writeln!(w, "{};{};{}", t.p.id, elev_scaled, iso).unwrap();
                        }
                        Err(e) => {
                            eprintln!("{};ERROR:{}", t.p.id, e);
                        }
                    }
                }
            });
        });

    out.lock().unwrap().flush()?;

    Ok(())
}

fn cluster_tasks(
    tasks: &[PeakTask],
    gt: [f64; 6],
    raster_size: (isize, isize),
    block_size: (isize, isize),
    cluster_shift: u8,
) -> Vec<Vec<PeakTask>> {
    let (bw, bh) = block_size;

    let mut map: HashMap<(isize, isize), Vec<PeakTask>> = HashMap::new();

    for t in tasks {
        let (px_f, py_f) = coord_to_pixel(gt, t.p.point);
        let x0 = px_f.round() as isize;
        let y0 = py_f.round() as isize;

        // Peaks outside DEM will still get a result (search_radius), but put them into a single cluster.
        let (bx, by) = if x0 < 0 || y0 < 0 || x0 >= raster_size.0 || y0 >= raster_size.1 {
            (0, 0)
        } else {
            (x0.div_euclid(bw), y0.div_euclid(bh))
        };

        let key = (bx >> cluster_shift, by >> cluster_shift);

        map.entry(key).or_default().push(*t);
    }

    let mut clusters = map
        .into_values()
        .map(|mut v| {
            v.sort_by(|a, b| {
                let (ax, ay) = coord_to_pixel(gt, a.p.point);
                let (bx, by) = coord_to_pixel(gt, b.p.point);

                let ax = ax.round() as isize;
                let ay = ay.round() as isize;
                let bx = bx.round() as isize;
                let by = by.round() as isize;

                (ay, ax).cmp(&(by, bx))
            });

            v
        })
        .collect::<Vec<_>>();

    clusters.sort_by_key(|b| std::cmp::Reverse(b.len()));

    clusters
}

fn read_peaks(
    file: &Path,
    proj: &Proj,
    band: &gdal::raster::RasterBand,
    gt: [f64; 6],
    raster_size: (isize, isize),
    block_size: (isize, isize),
    block_cache: usize,
) -> Result<Vec<Peak>> {
    let file = File::open(file).context("Error opening input file")?;

    let mut cache = BlockCache::<DataType>::new(block_cache);

    let nodata = band.no_data_value();

    let mut peaks = BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let mut it = line.split(';');

            let id: u64 = it
                .next()
                .ok_or(anyhow!("Expected id"))?
                .parse()
                .context("id must be u64")?;

            let point = Point::new(
                it.next()
                    .ok_or(anyhow!("Expected lon"))?
                    .parse()
                    .context("lon must be f64")?,
                it.next()
                    .ok_or(anyhow!("Expected lat"))?
                    .parse()
                    .context("lat must be f64")?,
            );

            // Optional input elevation (ignored; DEM elevation is used instead).
            let _ = it.next();

            let point = proj
                .convert(point)
                .context("Error projecting coordinates")?;

            let elev =
                elevation_from_dem(band, gt, raster_size, block_size, &mut cache, nodata, point)?
                    .unwrap_or(0);

            Ok(Peak { id, point, elev })
        })
        .collect::<Result<Vec<_>>>()?;

    peaks.retain(|p| p.elev > 0);

    Ok(peaks)
}

fn coord_to_pixel(gt: [f64; 6], point: Point) -> (f64, f64) {
    let x = (point.x() - gt[0]) / gt[1];
    let y = (point.y() - gt[3]) / gt[5];

    (x, y)
}

fn elevation_from_dem(
    band: &gdal::raster::RasterBand,
    gt: [f64; 6],
    raster_size: (isize, isize),
    block_size: (isize, isize),
    cache: &mut BlockCache<DataType>,
    nodata: Option<f64>,
    point: Point,
) -> Result<Option<DataType>> {
    let (px_f, py_f) = coord_to_pixel(gt, point);

    let x = px_f.round() as isize;
    let y = py_f.round() as isize;

    if x < 0 || y < 0 || x >= raster_size.0 || y >= raster_size.1 {
        return Ok(None);
    }

    let z = read_band_data_cached(band, block_size, cache, x, y)?;

    if let Some(nd) = nodata
        && (z as f64) == nd
    {
        return Ok(None);
    }

    Ok(Some(z))
}

fn pixel_center_to_lonlat(gt: [f64; 6], px: f64, py: f64) -> Point {
    let x = px + 0.5;
    let y = py + 0.5;

    let lon = gt[0] + x * gt[1] + y * gt[2];
    let lat = gt[3] + x * gt[4] + y * gt[5];

    Point::new(lon, lat)
}

fn approx_meters_per_pixel(gt: [f64; 6], lat: f64) -> f64 {
    let m_per_deg_lat = 111_320.0;
    let m_per_deg_lon = 111_320.0 * lat.to_radians().cos().abs();

    let mx = gt[1].abs() * m_per_deg_lon;
    let my = gt[5].abs() * m_per_deg_lat;

    mx.max(my).max(1e-9)
}

fn isolation_by_dem(
    state: &mut WorkerState,
    p: Peak,
    search_radius_m: f64,
    effective_min_m: f64,
) -> Result<f64> {
    let raster_size = state.raster_size;

    let gt = state.gt;

    let mut try_candidate = |x: isize, y: isize, best: f64| -> Result<f64> {
        if x < 0 || y < 0 || x >= raster_size.0 || y >= raster_size.1 {
            return Ok(best);
        }

        let band = state
            .ds
            .rasterband(1)
            .context("Error fetching raster band 1")?;

        let z = read_band_data_cached(&band, state.block_size, &mut state.cache, x, y)?;

        if z < p.elev {
            return Ok(best);
        }

        let point = pixel_center_to_lonlat(gt, x as f64, y as f64);

        let d = Euclidean.distance(p.point, point);

        if d < effective_min_m {
            return Ok(best);
        }

        Ok(best.min(d))
    };

    let (px_f, py_f) = coord_to_pixel(gt, p.point);

    let x0 = px_f.round() as isize;
    let y0 = py_f.round() as isize;

    if x0 < 0 || y0 < 0 || x0 >= raster_size.0 || y0 >= raster_size.1 {
        return Ok(search_radius_m);
    }

    let mpp = approx_meters_per_pixel(gt, p.point.y());
    let k_max = (search_radius_m / mpp).ceil() as isize;

    let mut best = search_radius_m;

    for k in 0..=k_max {
        if k > 0 && (k as f64) * mpp > best {
            break;
        }

        for dx in -k..=k {
            let x = x0 + dx;

            let y_top = y0 - k;
            let y_bot = y0 + k;

            if y_top >= 0 {
                best = try_candidate(x, y_top, best)?;
            }

            if k != 0 && y_bot < raster_size.1 {
                best = try_candidate(x, y_bot, best)?;
            }
        }

        for dy in (-k + 1)..=(k - 1) {
            if k == 0 {
                break;
            }

            let y = y0 + dy;

            let x_left = x0 - k;
            let x_right = x0 + k;

            best = try_candidate(x_left, y, best)?;

            best = try_candidate(x_right, y, best)?;
        }
    }

    Ok(best)
}

fn read_band_data_cached(
    band: &gdal::raster::RasterBand,
    block_size: (isize, isize),
    cache: &mut BlockCache<DataType>,
    x: isize,
    y: isize,
) -> Result<DataType> {
    let (bw, bh) = block_size;

    let bx = x.div_euclid(bw);
    let by = y.div_euclid(bh);

    let key = (bx, by);

    if !cache.contains(&key) {
        let x_off = bx * bw;
        let y_off = by * bh;

        let buf = band
            .read_as::<DataType>(
                (x_off, y_off),
                (bw as usize, bh as usize),
                (bw as usize, bh as usize),
                None,
            )
            .context("Error reading data from the band")?;

        cache.insert(key, buf.data().into());
    }

    let data = cache.get(&key).unwrap();

    let ix = (x - bx * bw) as usize;
    let iy = (y - by * bh) as usize;

    Ok(data[iy * (bw as usize) + ix])
}

// Simple block-level LRU cache for GDAL raster blocks.
// Keeps recently used blocks in memory to avoid repeated GDALRasterIO calls.
// Keyed by (block_x, block_y).
struct BlockCache<T> {
    cap: usize,
    map: HashMap<(isize, isize), Vec<T>>,
    order: VecDeque<(isize, isize)>,
}

impl<T> BlockCache<T> {
    fn new(cap: usize) -> Self {
        Self {
            cap: cap.max(1),
            map: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn contains(&self, k: &(isize, isize)) -> bool {
        self.map.contains_key(k)
    }

    fn get(&mut self, k: &(isize, isize)) -> Option<&Vec<T>> {
        if self.map.contains_key(k) {
            self.order.retain(|x| x != k);
            self.order.push_back(*k);
        }

        self.map.get(k)
    }

    fn insert(&mut self, k: (isize, isize), v: Vec<T>) {
        if self.map.contains_key(&k) {
            self.order.retain(|x| *x != k);
        }

        self.map.insert(k, v);
        self.order.push_back(k);

        while self.order.len() > self.cap {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
    }
}
