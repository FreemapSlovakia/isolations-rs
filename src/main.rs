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
//   id;lon;lat;elev;isolation_m;blocker_lon;blocker_lat (EPSG:4326)
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
    sync::atomic::{AtomicUsize, Ordering},
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

    /// Snap radius (meters). 0 disables snapping. Window never exceeds nearest-neighbor distance.
    #[arg(long, default_value_t = 30.0)]
    snap_radius: f64,

    /// Snap window growth factor when max is on edge.
    #[arg(long, default_value_t = 2.0)]
    snap_grow: f64,

    /// Number of snap attempts (initial + retries).
    #[arg(long, default_value_t = 3)]
    snap_tries: usize,

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
    nn_dist_m: f64,
}

#[derive(Clone, Copy, Debug)]
struct PeakTask {
    p: Peak,
    search_radius_m: f64,
    higher_peak: Option<Point>,
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
    nodata: Option<f64>,
    proj_to_wgs84: Proj,
}

thread_local! {
    static WORKER: RefCell<Option<WorkerState>> = const { RefCell::new(None) };
}

struct DemMeta {
    gt: [f64; 6],
    width: usize,
    height: usize,
    block_w: usize,
    block_h: usize,
    nodata: Option<f64>,
    scale: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let meta = {
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
        let nodata = band.no_data_value();

        DemMeta {
            gt,
            width,
            height,
            block_w,
            block_h,
            nodata,
            scale,
        }
    };

    let mut peaks_raw = read_peaks(&args.input).context("Error reading peaks")?;

    println!("Peaks (raw): {}", peaks_raw.len());

    if peaks_raw.is_empty() {
        return Ok(());
    }

    // Compute per-peak nearest-neighbor distance for snap cap.
    let mut snap_tree = RTree::new();

    for peak in &peaks_raw {
        snap_tree.insert(*peak);
    }

    for peak in &mut peaks_raw {
        let mut nn = None;

        for (other, distance2) in snap_tree.nearest_neighbor_iter_with_distance_2(&peak.point) {
            if other.id == peak.id {
                continue;
            }

            nn = Some(distance2.sqrt());

            break;
        }

        peak.nn_dist_m = nn.unwrap_or(0.0);
    }

    let peak_clusters = cluster_peaks(&peaks_raw, &meta, args.cluster_shift);

    let threads = args
        .threads
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(NonZero::get)
                .unwrap_or(1)
        })
        .max(1);

    let peaks = fill_elevations(peak_clusters, &args.dem, &meta, threads, &args)
    .context("Error filling elevations from DEM")?;

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
            for (peak, distance2) in tree.nearest_neighbor_iter_with_distance_2(&p.point) {
                if peak.id == p.id {
                    continue;
                }

                if peak.elev > p.elev || distance2 > max_radius2 {
                    let search_radius_m = distance2.sqrt().min(args.max_radius);

                    cnt += 1;

                    return PeakTask {
                        p: *p,
                        search_radius_m,
                        higher_peak: if peak.elev > p.elev {
                            Some(peak.point)
                        } else {
                            None
                        },
                    };
                }
            }

            cnt += 1;

            PeakTask {
                p: *p,
                search_radius_m: args.max_radius,
                higher_peak: None,
            }
        })
        .collect::<Vec<_>>();

    println!("Tasks: {}", tasks.len());

    // Cluster tasks by super-block for cache locality.
    let clusters = cluster_tasks(&tasks, &meta, args.cluster_shift);

    println!("Clusters: {}", clusters.len());

    // --- Parallel execution using rayon ---
    // Each rayon worker keeps its own GDAL Dataset + BlockCache via thread-local storage.

    let total_tasks = tasks.len().max(1);
    let done = Arc::new(AtomicUsize::new(0));
    let last_percent = Arc::new(AtomicUsize::new(0));

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
                                raster_size: (meta.width as isize, meta.height as isize),
                                gt: meta.gt,
                                block_size: (meta.block_w as isize, meta.block_h as isize),
                                cache: BlockCache::<DataType>::new(args.block_cache),
                                nodata: meta.nodata,
                                proj_to_wgs84: Proj::new_known_crs("EPSG:3035", "EPSG:4326", None)
                                    .expect("proj EPSG:3035->4326"),
                            });
                        }

                        let mut borrow = cell.borrow_mut();

                        let state = borrow.as_mut().unwrap();

                        isolation_by_dem(state, t.p, t.search_radius_m, t.higher_peak)
                    });

                    match iso {
                        Ok((iso, blocker)) => {
                            let mut w = out.lock().unwrap();
                            let elev_scaled = (t.p.elev as f64) * meta.scale;
                            let peak_ll = WORKER.with(|cell| {
                                cell.borrow()
                                    .as_ref()
                                    .unwrap()
                                    .proj_to_wgs84
                                    .convert(t.p.point)
                            });
                            let peak_ll = peak_ll.expect("Error projecting peak to EPSG:4326");
                            if let Some(b) = blocker {
                                writeln!(
                                    w,
                                    "{};{};{};{};{};{};{}",
                                    t.p.id,
                                    peak_ll.x(),
                                    peak_ll.y(),
                                    elev_scaled,
                                    iso,
                                    b.x(),
                                    b.y()
                                )
                                .unwrap();
                            } else {
                                writeln!(
                                    w,
                                    "{};{};{};{};{};;",
                                    t.p.id,
                                    peak_ll.x(),
                                    peak_ll.y(),
                                    elev_scaled,
                                    iso
                                )
                                .unwrap();
                            }
                        }
                        Err(e) => {
                            eprintln!("{};ERROR:{}", t.p.id, e);
                        }
                    }

                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                    let pct = ((n * 100) / total_tasks).min(100);
                    let prev = last_percent.load(Ordering::Relaxed);
                    if pct != prev
                        && last_percent
                            .compare_exchange(prev, pct, Ordering::Relaxed, Ordering::Relaxed)
                            .is_ok()
                    {
                        eprintln!("Progress: {pct}%");
                    }
                }
            });
        });

    out.lock().unwrap().flush()?;

    Ok(())
}

fn cluster_tasks(tasks: &[PeakTask], meta: &DemMeta, cluster_shift: u8) -> Vec<Vec<PeakTask>> {
    let (bw, bh) = (meta.block_w as isize, meta.block_h as isize);

    let mut map: HashMap<(isize, isize), Vec<PeakTask>> = HashMap::new();

    for t in tasks {
        let (px_f, py_f) = coord_to_pixel(meta.gt, t.p.point);
        let x0 = px_f.round() as isize;
        let y0 = py_f.round() as isize;

        // Peaks outside DEM will still get a result (search_radius), but put them into a single cluster.
        let (bx, by) =
            if x0 < 0 || y0 < 0 || x0 >= meta.width as isize || y0 >= meta.height as isize {
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
                let (ax, ay) = coord_to_pixel(meta.gt, a.p.point);
                let (bx, by) = coord_to_pixel(meta.gt, b.p.point);

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

fn cluster_peaks(peaks: &[Peak], meta: &DemMeta, cluster_shift: u8) -> Vec<Vec<Peak>> {
    let (bw, bh) = (meta.block_w as isize, meta.block_h as isize);

    let mut map: HashMap<(isize, isize), Vec<Peak>> = HashMap::new();

    for p in peaks {
        let (px_f, py_f) = coord_to_pixel(meta.gt, p.point);
        let x0 = px_f.round() as isize;
        let y0 = py_f.round() as isize;

        // Peaks outside DEM will still get a result later, but group them in a single cluster.
        let (bx, by) =
            if x0 < 0 || y0 < 0 || x0 >= meta.width as isize || y0 >= meta.height as isize {
                (0, 0)
            } else {
                (x0.div_euclid(bw), y0.div_euclid(bh))
            };

        let key = (bx >> cluster_shift, by >> cluster_shift);

        map.entry(key).or_default().push(*p);
    }

    let mut clusters = map
        .into_values()
        .map(|mut v| {
            v.sort_by(|a, b| {
                let (ax, ay) = coord_to_pixel(meta.gt, a.point);
                let (bx, by) = coord_to_pixel(meta.gt, b.point);

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

fn fill_elevations(
    peak_clusters: Vec<Vec<Peak>>,
    dem: &Path,
    meta: &DemMeta,
    threads: usize,
    args: &Args,
) -> Result<Vec<Peak>> {
    let filled = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .context("Error building thread pool")?
        .install(|| {
            peak_clusters
                .into_par_iter()
                .map(|cluster| {
                    WORKER.with(|cell| {
                        if cell.borrow().is_none() {
                            *cell.borrow_mut() = Some(WorkerState {
                                ds: Dataset::open(dem).expect("open DEM"),
                                raster_size: (meta.width as isize, meta.height as isize),
                                gt: meta.gt,
                                block_size: (meta.block_w as isize, meta.block_h as isize),
                                cache: BlockCache::<DataType>::new(args.block_cache),
                                nodata: meta.nodata,
                                proj_to_wgs84: Proj::new_known_crs("EPSG:3035", "EPSG:4326", None)
                                    .expect("proj EPSG:3035->4326"),
                            });
                        }

                        let mut borrow = cell.borrow_mut();

                        let state = borrow.as_mut().unwrap();

                        let mut out = Vec::with_capacity(cluster.len());

                        for p in cluster {
                            if let Some((point, elev)) = snap_peak_in_window(
                                state,
                                p.point,
                                p.nn_dist_m,
                                args.snap_radius,
                                args.snap_grow,
                                args.snap_tries,
                            )? && elev > 0
                            {
                                out.push(Peak {
                                    id: p.id,
                                    point,
                                    elev,
                                    nn_dist_m: p.nn_dist_m,
                                });
                            }
                        }

                        Ok(out)
                    })
                })
                .collect::<Vec<Result<Vec<Peak>>>>()
        });

    let mut peaks = Vec::new();

    for r in filled {
        let mut v = r?;
        peaks.append(&mut v);
    }

    Ok(peaks)
}

fn read_peaks(file: &Path) -> Result<Vec<Peak>> {
    let file = File::open(file).context("Error opening input file")?;

    let proj =
        Proj::new_known_crs("EPSG:4326", "EPSG:3035", None).context("Error creating projection")?;

    BufReader::new(file)
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

            Ok(Peak {
                id,
                point,
                elev: 0,
                nn_dist_m: 0.0,
            })
        })
        .collect()
}

fn coord_to_pixel(gt: [f64; 6], point: Point) -> (f64, f64) {
    let x = (point.x() - gt[0]) / gt[1];
    let y = (point.y() - gt[3]) / gt[5];

    (x, y)
}

fn read_elev_at(state: &mut WorkerState, x: isize, y: isize) -> Result<Option<DataType>> {
    if x < 0 || y < 0 || x >= state.raster_size.0 || y >= state.raster_size.1 {
        return Ok(None);
    }

    let band = state
        .ds
        .rasterband(1)
        .context("Error fetching raster band 1")?;

    let z = read_band_data_cached(&band, state.block_size, &mut state.cache, x, y)?;

    if let Some(nd) = state.nodata
        && (z as f64) == nd
    {
        return Ok(None);
    }

    Ok(Some(z))
}

fn snap_peak_in_window(
    state: &mut WorkerState,
    point: Point,
    nn_dist_m: f64,
    snap_cap_m: f64,
    snap_grow: f64,
    snap_tries: usize,
) -> Result<Option<(Point, DataType)>> {
    let (px_f, py_f) = coord_to_pixel(state.gt, point);
    let x0 = px_f.round() as isize;
    let y0 = py_f.round() as isize;

    let center = match read_elev_at(state, x0, y0)? {
        Some(v) => v,
        None => return Ok(None),
    };

    if snap_cap_m <= 0.0 || nn_dist_m <= 0.0 {
        return Ok(Some((point, center)));
    }

    let mpp = state.gt[1].abs().max(state.gt[5].abs()).max(1e-9);

    let tries = snap_tries.max(1);

    let mut radius = snap_cap_m;

    for _ in 0..tries {
        if radius > nn_dist_m {
            return Ok(Some((point, center)));
        }

        let k = (radius / mpp).ceil() as isize;

        if k <= 0 {
            return Ok(Some((point, center)));
        }

        let mut max = center;
        let mut max_pos = (x0, y0);
        let mut max_on_edge = false;

        for dy in -k..=k {
            let y = y0 + dy;
            for dx in -k..=k {
                let x = x0 + dx;
                let z = match read_elev_at(state, x, y)? {
                    Some(v) => v,
                    None => continue,
                };

                if z > max {
                    max = z;
                    max_pos = (x, y);
                    max_on_edge = dx.abs() == k || dy.abs() == k;
                }
            }
        }

        if max > center && !max_on_edge {
            let snapped = pixel_center_to_lonlat(state.gt, max_pos.0 as f64, max_pos.1 as f64);
            return Ok(Some((snapped, max)));
        }

        if snap_grow <= 1.0 {
            break;
        }

        radius *= snap_grow;
    }

    Ok(Some((point, center)))
}

fn pixel_center_to_lonlat(gt: [f64; 6], px: f64, py: f64) -> Point {
    let x = px + 0.5;
    let y = py + 0.5;

    let lon = gt[0] + x * gt[1] + y * gt[2];
    let lat = gt[3] + x * gt[4] + y * gt[5];

    Point::new(lon, lat)
}

fn isolation_by_dem(
    state: &mut WorkerState,
    peak: Peak,
    search_radius_m: f64,
    higher_peak: Option<Point>,
) -> Result<(f64, Option<Point>)> {
    let raster_size = state.raster_size;

    let gt = state.gt;

    let mut best_point: Option<Point> = None;

    let mut try_candidate =
        |x: isize, y: isize, best: f64, best_point: &mut Option<Point>| -> Result<f64> {
            if x < 0 || y < 0 || x >= raster_size.0 || y >= raster_size.1 {
                return Ok(best);
            }

            let band = state
                .ds
                .rasterband(1)
                .context("Error fetching raster band 1")?;

            let z = read_band_data_cached(&band, state.block_size, &mut state.cache, x, y)?;

            if z <= peak.elev {
                return Ok(best);
            }

            let point = pixel_center_to_lonlat(gt, x as f64, y as f64);

            let d = Euclidean.distance(peak.point, point);

            if d < best {
                *best_point = Some(point);
                return Ok(d);
            }

            Ok(best)
        };

    let (px_f, py_f) = coord_to_pixel(gt, peak.point);

    let x0 = px_f.round() as isize;
    let y0 = py_f.round() as isize;

    if x0 < 0 || y0 < 0 || x0 >= raster_size.0 || y0 >= raster_size.1 {
        return Ok((search_radius_m, None));
    }

    let mpp = gt[1].abs().max(gt[5].abs());

    let k_max = (search_radius_m / mpp).ceil() as isize;

    let mut best = search_radius_m;

    for k in 0..=k_max {
        if k > 0 && (k as f64) * mpp > best {
            break;
        }

        for d in -k..k {
            best = try_candidate(x0 + d, y0 - k, best, &mut best_point)?;
            best = try_candidate(x0 + k, y0 + d, best, &mut best_point)?;
            best = try_candidate(x0 - d, y0 + k, best, &mut best_point)?;
            best = try_candidate(x0 - k, y0 - d, best, &mut best_point)?;
        }
    }

    let best_point = match best_point.or(higher_peak) {
        Some(p) => Some(
            state
                .proj_to_wgs84
                .convert(p)
                .context("Error projecting blocker to EPSG:4326")?,
        ),
        None => None,
    };

    Ok((best, best_point))
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
