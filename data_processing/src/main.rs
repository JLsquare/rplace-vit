use chrono::NaiveDateTime;
use flate2::read::GzDecoder;
use image::{ImageBuffer, Rgb};
use indicatif::{ProgressBar, ProgressStyle};
use log::{error, info, warn};
use rayon::prelude::*;
use rusqlite::{params, Connection, Result};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::fs::File;
use std::io::copy;
use std::io::Write;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::{Arc, Mutex};
use thiserror::Error;

#[derive(Error, Debug)]
enum RPlaceDatasetError {
    #[error("Database Error: {0}")]
    DatabaseError(#[from] rusqlite::Error),

    #[error("IO Error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Invalid Color Error: {0}")]
    InvalidColorError(String),

    #[error("Malformed Line: {0}")]
    MalformedLineError(String),

    #[error("Timestamp Parsing Error: {0}")]
    TimestampParsingError(#[from] chrono::ParseError),

    #[error("Image Error: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("Reqwest Error: {0}")]
    ReqwestError(#[from] reqwest::Error),
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env::set_var("RUST_LOG", "info");
    env_logger::init();

    let mut mode = String::new();
    println!(
        "Enter the mode (download, unsorted, sorted, partition, end_states, all, png (bonus)): "
    );
    std::io::stdin().read_line(&mut mode).unwrap();
    let mode = mode.trim();

    match mode {
        "download" => download_dataset()?,
        "unsorted" => {
            let canvas_size = get_canvas_size();
            process_unsorted(canvas_size)?
        }
        "sorted" => process_sorted()?,
        "partition" => partition_sorted_db()?,
        "end_states" => process_end_states()?,
        "png" => process_end_states_to_png()?,
        "all" => {
            let canvas_size = get_canvas_size();
            info!("Starting data processing...");
            download_dataset()?;
            process_unsorted(canvas_size)?;
            process_sorted()?;
            partition_sorted_db()?;
            process_end_states()?;
            info!("All data processing completed successfully.");
        }
        _ => {
            error!("Invalid mode. Use 'download', 'unsorted', 'sorted', 'partition', 'end_states', 'all', or 'png'");
            std::process::exit(1);
        }
    }

    Ok(())
}

fn get_canvas_size() -> i64 {
    let mut canvas_size = String::new();
    println!("Enter the canvas size (2000): ");
    std::io::stdin().read_line(&mut canvas_size).unwrap();
    let canvas_size = canvas_size.trim();
    let canvas_size = if canvas_size.is_empty() {
        "2000"
    } else {
        canvas_size
    };
    canvas_size.parse::<i64>().unwrap()
}

fn create_progress_bar(len: u64, download: bool) -> ProgressBar {
    let pb = ProgressBar::new(len);

    let template = if download {
        "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})"
    } else {
        "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})"
    };

    pb.set_style(
        ProgressStyle::default_bar()
            .template(template)
            .unwrap()
            .progress_chars("#>-"),
    );
    pb
}

fn download_dataset() -> Result<(), Box<dyn std::error::Error>> {
    info!("Downloading dataset...");
    let url = "https://placedata.reddit.com/data/canvas-history/2022_place_canvas_history.csv.gzip";
    let resp = reqwest::blocking::get(url)?;
    let total_size = resp.content_length().unwrap_or(0);

    let pb = create_progress_bar(total_size, true);

    let mut source = pb.wrap_read(resp);

    let mut compressed_file = File::create("2022_place_canvas_history.csv.gzip")?;
    copy(&mut source, &mut compressed_file)?;

    pb.finish_with_message("Download completed");

    println!("Extracting file...");
    let compressed_file = File::open("2022_place_canvas_history.csv.gzip")?;
    let mut decoder = GzDecoder::new(compressed_file);
    let mut extracted_file = File::create("data.csv")?;
    copy(&mut decoder, &mut extracted_file)?;

    std::fs::remove_file("2022_place_canvas_history.csv.gzip")?;

    println!("Dataset downloaded and extracted successfully.");
    Ok(())
}

fn process_unsorted(canvas_size: i64) -> Result<(), RPlaceDatasetError> {
    info!("Creating unsorted database...");
    let conn = Arc::new(Mutex::new(Connection::open("unsorted.db")?));

    {
        let conn = conn.lock().unwrap();
        conn.execute(
            "CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_id TEXT UNIQUE
            )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS pixels (
                timestamp INTEGER,
                user_id INTEGER,
                color_id INTEGER,
                x INTEGER,
                y INTEGER,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )",
            [],
        )?;
    }

    let color_mapping = Arc::new(create_color_str_mapping());

    let file = File::open("data.csv")?;

    let reader = BufReader::new(file);
    let total_lines = reader.lines().count() as u64;
    let pb = create_progress_bar(total_lines, false);

    let reader = BufReader::new(File::open("data.csv")?);

    let chunk_size = 100000;
    let chunks: Vec<_> = reader
        .lines()
        .collect::<Result<Vec<_>, _>>()?
        .chunks(chunk_size)
        .map(|c| c.to_vec())
        .collect();

    chunks.par_iter().try_for_each(|chunk| {
        let conn = Arc::clone(&conn);
        let color_mapping = Arc::clone(&color_mapping);
        let result = process_chunk(&conn, chunk, &color_mapping, canvas_size);
        pb.inc(chunk.len() as u64);
        result
    })?;

    pb.finish_with_message("Unsorted database created successfully");
    Ok(())
}

fn process_chunk(
    conn: &Arc<Mutex<Connection>>,
    chunk: &[String],
    color_mapping: &HashMap<&str, i64>,
    canvas_size: i64,
) -> Result<(), RPlaceDatasetError> {
    let mut conn = conn.lock().unwrap();
    let tx = conn.transaction()?;

    let user_sql = "INSERT OR IGNORE INTO users (original_id) VALUES (?)";
    let pixel_sql = "INSERT INTO pixels (timestamp, user_id, color_id, x, y) VALUES (?, (SELECT id FROM users WHERE original_id = ?), ?, ?, ?)";

    {
        let mut user_stmt = tx.prepare_cached(user_sql)?;
        let mut pixel_stmt = tx.prepare_cached(pixel_sql)?;

        for (i, line) in chunk.iter().enumerate() {
            if i == 0 && chunk[0].starts_with("timestamp") {
                continue;
            }

            let fields: Vec<&str> = line.trim().split(',').collect();

            if fields.len() < 5 {
                warn!("Skipping malformed line: {}", line);
                continue;
            }

            let x: i64 = fields[3]
                .trim_matches('"')
                .parse()
                .map_err(|_| RPlaceDatasetError::MalformedLineError(line.to_string()))?;
            let y: i64 = fields[4]
                .trim_matches('"')
                .parse()
                .map_err(|_| RPlaceDatasetError::MalformedLineError(line.to_string()))?;

            if x < 0 || x >= canvas_size || y < 0 || y >= canvas_size {
                continue;
            }

            let timestamp =
                match NaiveDateTime::parse_from_str(fields[0], "%Y-%m-%d %H:%M:%S%.3f UTC") {
                    Ok(dt) => dt.and_utc().timestamp(),
                    Err(_) => {
                        let dt = NaiveDateTime::parse_from_str(fields[0], "%Y-%m-%d %H:%M:%S UTC")?;
                        dt.and_utc().timestamp()
                    }
                };

            let color_id = *color_mapping
                .get(fields[2])
                .ok_or_else(|| RPlaceDatasetError::InvalidColorError(fields[2].to_string()))?;

            user_stmt.execute(params![fields[1]])?;
            pixel_stmt.execute(params![timestamp, fields[1], color_id, x, y])?;
        }
    }

    tx.commit()?;
    Ok(())
}

fn process_sorted() -> Result<(), RPlaceDatasetError> {
    info!("Creating sorted database...");
    let src_conn = Connection::open("unsorted.db")?;
    let mut dest_conn = Connection::open("sorted.db")?;

    dest_conn.execute(
        "CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_id TEXT UNIQUE
        )",
        [],
    )?;

    dest_conn.execute(
        "CREATE TABLE IF NOT EXISTS pixels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER,
            user_id INTEGER,
            color_id INTEGER,
            x INTEGER,
            y INTEGER,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )",
        [],
    )?;

    let mut user_stmt = src_conn.prepare("SELECT id, original_id FROM users")?;
    let user_rows = user_stmt.query_map([], |row| {
        Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
    })?;

    let tx = dest_conn.transaction()?;
    {
        let mut insert_user_stmt =
            tx.prepare("INSERT INTO users (id, original_id) VALUES (?, ?)")?;
        for user_row in user_rows {
            let (id, original_id) = user_row?;
            insert_user_stmt.execute(params![id, original_id])?;
        }
    }
    tx.commit()?;

    let mut stmt = src_conn.prepare("SELECT COUNT(*) FROM pixels")?;
    let total_pixels: i64 = stmt.query_row([], |row| row.get(0))?;
    let pb = create_progress_bar(total_pixels as u64, false);

    let mut stmt = src_conn
        .prepare("SELECT timestamp, user_id, color_id, x, y FROM pixels ORDER BY timestamp ASC")?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, i64>(1)?,
            row.get::<_, i64>(2)?,
            row.get::<_, i64>(3)?,
            row.get::<_, i64>(4)?,
        ))
    })?;

    let tx = dest_conn.transaction()?;
    {
        let mut insert_stmt = tx.prepare(
            "INSERT INTO pixels (timestamp, user_id, color_id, x, y) VALUES (?, ?, ?, ?, ?)",
        )?;

        for row in rows {
            let (timestamp, user_id, color_id, x, y) = row?;
            insert_stmt.execute([timestamp, user_id, color_id, x, y])?;
            pb.inc(1);
        }
    }
    tx.commit()?;

    pb.finish_with_message("Sorted database created successfully");
    Ok(())
}

fn partition_sorted_db() -> Result<(), RPlaceDatasetError> {
    info!("Partitioning sorted database...");
    let src_conn = Connection::open("sorted.db")?;
    let partition_size = 1_000_000;
    let partition_folder = "partitions";

    fs::create_dir_all(partition_folder)?;

    let mut stmt = src_conn.prepare("SELECT COUNT(*) FROM pixels")?;
    let total_pixels: i64 = stmt.query_row([], |row| row.get(0))?;
    let num_partitions = (total_pixels as f64 / partition_size as f64).ceil() as i64;

    let pb = create_progress_bar(num_partitions as u64, false);

    (0..num_partitions).into_par_iter().try_for_each(|partition| {
        let db_name = format!("{}/partition_{}.db", partition_folder, partition);
        let mut dest_conn = Connection::open(&db_name)?;

        dest_conn.execute(
            "CREATE TABLE IF NOT EXISTS pixels (
                id INTEGER PRIMARY KEY,
                timestamp INTEGER,
                user_id INTEGER,
                color_id INTEGER,
                x INTEGER,
                y INTEGER
            )",
            [],
        )?;

        dest_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_timestamp ON pixels (timestamp)",
            [],
        )?;

        dest_conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_coordinates ON pixels (x, y)",
            [],
        )?;

        let start_id = partition * partition_size;
        let end_id = (partition + 1) * partition_size;

        let src_conn = Connection::open("sorted.db")?;
        let mut src_stmt = src_conn.prepare(
            "SELECT pixels.id, pixels.timestamp, pixels.user_id, pixels.color_id, pixels.x, pixels.y 
             FROM pixels 
             WHERE pixels.id >= ? AND pixels.id < ? 
             ORDER BY pixels.id ASC"
        )?;

        let rows = src_stmt.query_map([start_id, end_id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, i64>(5)?,
            ))
        })?;

        let tx = dest_conn.transaction()?;
        {
            let mut insert_stmt = tx.prepare(
                "INSERT INTO pixels (id, timestamp, user_id, color_id, x, y) VALUES (?, ?, ?, ?, ?, ?)"
            )?;

            for row in rows {
                let (id, timestamp, user_id, color_id, x, y) = row?;
                insert_stmt.execute(params![id, timestamp, user_id, color_id, x, y])?;
            }
        }
        tx.commit()?;

        pb.inc(1);
        Ok::<(), RPlaceDatasetError>(())
    })?;

    pb.finish_with_message("Database partitioning complete");
    Ok(())
}

fn process_end_states() -> Result<(), RPlaceDatasetError> {
    info!("Creating end states...");
    let partition_folder = "partitions";
    let output_folder = "end_states";
    let canvas_size = 2000;

    fs::create_dir_all(output_folder)?;

    let mut end_state = vec![vec![31; canvas_size]; canvas_size];
    let mut partition_count = 0;

    while std::path::Path::new(&format!(
        "{}/partition_{}.db",
        partition_folder, partition_count
    ))
    .exists()
    {
        partition_count += 1;
    }

    let pb = create_progress_bar(partition_count as u64, false);

    for partition in 0.. {
        let db_name = format!("{}/partition_{}.db", partition_folder, partition);

        if !std::path::Path::new(&db_name).exists() {
            break;
        }

        let conn = Connection::open(&db_name)?;

        let mut stmt = conn.prepare("SELECT x, y, color_id FROM pixels ORDER BY timestamp ASC")?;

        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })?;

        for row in rows {
            let (x, y, color_id) = row?;
            if x >= 0 && x < canvas_size as i64 && y >= 0 && y < canvas_size as i64 {
                end_state[y as usize][x as usize] = color_id;
            }
        }

        let output_file = format!("{}/end_state_{}.txt", output_folder, partition);
        let mut file = File::create(output_file)?;

        for row in &end_state {
            let line: String = row
                .iter()
                .map(|&color_id| color_id.to_string())
                .collect::<Vec<String>>()
                .join(";");
            writeln!(file, "{}", line)?;
        }

        pb.inc(1);
    }

    pb.finish_with_message("End states created successfully");
    Ok(())
}

fn process_end_states_to_png() -> Result<(), RPlaceDatasetError> {
    info!("Creating PNG files from end states...");
    let input_folder = "end_states";
    let output_folder = "png_output";
    let canvas_size = 2000;

    std::fs::create_dir_all(output_folder)?;

    let color_mapping = create_color_rgb_mapping();

    let partition_count = fs::read_dir(input_folder)?.count();
    let pb = create_progress_bar(partition_count as u64, false);

    for entry in std::fs::read_dir(input_folder)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("txt") {
            let file_name = path.file_stem().and_then(|s| s.to_str()).unwrap();
            let output_path = Path::new(output_folder).join(format!("{}.png", file_name));

            info!("Processing file: {:?}", path);

            let mut img = ImageBuffer::new(canvas_size as u32, canvas_size as u32);

            let file = File::open(&path)?;
            let reader = BufReader::new(file);

            for (y, line) in reader.lines().enumerate() {
                let line = line?;
                let color_ids: Vec<i64> = line.split(';').filter_map(|s| s.parse().ok()).collect();

                for (x, &color_id) in color_ids.iter().enumerate() {
                    if x < canvas_size && y < canvas_size {
                        let color = color_mapping.get(&color_id).unwrap_or(&[0, 0, 0]);
                        img.put_pixel(x as u32, y as u32, Rgb(*color));
                    }
                }
            }

            img.save(&output_path)?;
            pb.inc(1);
        }
    }

    pb.finish_with_message("PNG files created successfully");
    Ok(())
}

fn create_color_rgb_mapping() -> HashMap<i64, [u8; 3]> {
    [
        (0, [109, 0, 26]),
        (1, [190, 0, 57]),
        (2, [255, 69, 0]),
        (3, [255, 168, 0]),
        (4, [255, 214, 53]),
        (5, [255, 248, 184]),
        (6, [0, 163, 104]),
        (7, [0, 204, 120]),
        (8, [126, 237, 86]),
        (9, [0, 117, 111]),
        (10, [0, 158, 170]),
        (11, [0, 204, 192]),
        (12, [36, 80, 164]),
        (13, [54, 144, 234]),
        (14, [81, 233, 244]),
        (15, [73, 58, 193]),
        (16, [106, 92, 255]),
        (17, [148, 179, 255]),
        (18, [129, 30, 159]),
        (19, [180, 74, 192]),
        (20, [228, 171, 255]),
        (21, [222, 16, 127]),
        (22, [255, 56, 129]),
        (23, [255, 153, 170]),
        (24, [109, 72, 47]),
        (25, [156, 105, 38]),
        (26, [255, 180, 112]),
        (27, [0, 0, 0]),
        (28, [81, 82, 82]),
        (29, [137, 141, 144]),
        (30, [212, 215, 217]),
        (31, [255, 255, 255]),
    ]
    .iter()
    .cloned()
    .collect()
}

fn create_color_str_mapping() -> HashMap<&'static str, i64> {
    [
        ("#6D001A", 0),
        ("#BE0039", 1),
        ("#FF4500", 2),
        ("#FFA800", 3),
        ("#FFD635", 4),
        ("#FFF8B8", 5),
        ("#00A368", 6),
        ("#00CC78", 7),
        ("#7EED56", 8),
        ("#00756F", 9),
        ("#009EAA", 10),
        ("#00CCC0", 11),
        ("#2450A4", 12),
        ("#3690EA", 13),
        ("#51E9F4", 14),
        ("#493AC1", 15),
        ("#6A5CFF", 16),
        ("#94B3FF", 17),
        ("#811E9F", 18),
        ("#B44AC0", 19),
        ("#E4ABFF", 20),
        ("#DE107F", 21),
        ("#FF3881", 22),
        ("#FF99AA", 23),
        ("#6D482F", 24),
        ("#9C6926", 25),
        ("#FFB470", 26),
        ("#000000", 27),
        ("#515252", 28),
        ("#898D90", 29),
        ("#D4D7D9", 30),
        ("#FFFFFF", 31),
    ]
    .iter()
    .cloned()
    .collect()
}
