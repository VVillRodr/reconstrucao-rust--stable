use common::ReconstructionRequest;
use ndarray::Array1;
use reqwest::Client;
use std::fmt;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use uuid::Uuid;

const SERVER_URL: &str = "http://127.0.0.1:3000";

fn resolve_csv_path(file_path: &str) -> PathBuf {
    let workspace_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();

    workspace_root.join(file_path)
}

fn read_g_vector_from_csv(file_path: &str) -> Result<Array1<f64>, String> {
    let full_path = resolve_csv_path(file_path);

    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path(&full_path)
        .map_err(|e| format!("{} (path: {})", e.to_string(), full_path.display()))?;

    let mut data = Vec::new();
    for result in reader.records() {
        let record = result.map_err(|e| e.to_string())?;
        let value: f64 = record[0].trim().parse().map_err(|e: std::num::ParseFloatError| e.to_string())?;
        data.push(value);
    }
    Ok(Array1::from(data))
}

struct ReportEntry {
    signal_file: String,
    algorithm: String,
    model_id: String,
    time_secs: f64,
    iterations: usize,
    status: String,
}

impl fmt::Display for ReportEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:<20} {:<6} {:<6} {:>8.4}s {:>10} {}",
            self.signal_file,
            self.algorithm,
            self.model_id,
            self.time_secs,
            self.iterations,
            self.status
        )
    }
}

#[tokio::main]
async fn main() {
    let client = Client::new();
    let signal_files = [
        "g-30x30-1.csv",
        "g-30x30-2.csv",
        "g-30x30-3.csv",
        "G-60x60-1.csv",
        "G-60x60-2.csv",
        "G-60x60-3.csv",
    ];
    let algorithms = ["CGNR", "CGNE"];

    let mut report_rows = Vec::new();

    for signal_file in signal_files.iter() {
        println!("[Report] Processando arquivo {}", signal_file);

        let g_vector = match read_g_vector_from_csv(signal_file) {
            Ok(g) => g,
            Err(err) => {
                eprintln!("[Report] Erro ao ler {}: {}", signal_file, err);
                continue;
            }
        };

        let model_id = if signal_file.to_lowercase().contains("60x60") {
            "60x60"
        } else {
            "30x30"
        };

        for algorithm in algorithms.iter() {
            let request = ReconstructionRequest {
                user_id: Uuid::new_v4(),
                algorithm_id: algorithm.to_string(),
                model_id: model_id.to_string(),
                g: g_vector.to_vec(),
            };

            println!("[Report] Enviando {} para {}...", algorithm, signal_file);
            let started = Instant::now();
            let response = client
                .post(format!("{}/reconstruct", SERVER_URL))
                .json(&request)
                .send()
                .await;
            let elapsed = started.elapsed();

            match response {
                Ok(resp) => {
                    let status_code = resp.status();
                    if status_code.is_success() {
                        match resp.json::<common::ReconstructionResult>().await {
                            Ok(result) => {
                                report_rows.push(ReportEntry {
                                    signal_file: signal_file.to_string(),
                                    algorithm: algorithm.to_string(),
                                    model_id: model_id.to_string(),
                                    time_secs: result.reconstruction_time_ms as f64 / 1000.0,
                                    iterations: result.iterations,
                                    status: "OK".to_string(),
                                });
                                println!(
                                    "[Report] {} {} concluído em {:.3}s (iter: {}).",
                                    signal_file,
                                    algorithm,
                                    result.reconstruction_time_ms as f64 / 1000.0,
                                    result.iterations
                                );
                            }
                            Err(err) => {
                                eprintln!("[Report] Falha ao desserializar resposta: {}", err);
                                report_rows.push(ReportEntry {
                                    signal_file: signal_file.to_string(),
                                    algorithm: algorithm.to_string(),
                                    model_id: model_id.to_string(),
                                    time_secs: elapsed.as_secs_f64(),
                                    iterations: 0,
                                    status: format!("parse_error:{}", err),
                                });
                            }
                        }
                    } else {
                        let text = resp.text().await.unwrap_or_default();
                        eprintln!("[Report] Erro HTTP {}: {}", status_code, text);
                        report_rows.push(ReportEntry {
                            signal_file: signal_file.to_string(),
                            algorithm: algorithm.to_string(),
                            model_id: model_id.to_string(),
                            time_secs: elapsed.as_secs_f64(),
                            iterations: 0,
                            status: format!("http_error:{}", status_code),
                        });
                    }
                }
                Err(err) => {
                    eprintln!("[Report] Falha na requisição {} {}: {}", signal_file, algorithm, err);
                    report_rows.push(ReportEntry {
                        signal_file: signal_file.to_string(),
                        algorithm: algorithm.to_string(),
                        model_id: model_id.to_string(),
                        time_secs: elapsed.as_secs_f64(),
                        iterations: 0,
                        status: format!("request_error:{}", err),
                    });
                }
            }
        }
    }

    println!("\n--- Relatório Final ---");
    println!("{:<20} {:<6} {:<6} {:>10} {:>12} {}", "Sinal", "Algo", "Modelo", "Tempo(s)", "Iterações", "Status");
    println!("{:-<80}", "");
    for row in report_rows.iter() {
        println!("{}", row);
    }

    if let Err(err) = save_report_csv(&report_rows) {
        eprintln!("[Report] Erro ao salvar relatório CSV: {}", err);
    } else {
        println!("[Report] Relatório salvo em report_results.csv");
    }
}

fn save_report_csv(rows: &[ReportEntry]) -> Result<(), String> {
    let path = resolve_csv_path("report_results.csv");
    let mut file = File::create(&path).map_err(|e| e.to_string())?;
    writeln!(file, "signal_file,algorithm,model_id,time_secs,iterations,status")
        .map_err(|e| e.to_string())?;
    for row in rows {
        writeln!(
            file,
            "{},{},{},{:.6},{},{}",
            row.signal_file, row.algorithm, row.model_id, row.time_secs, row.iterations, row.status
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}
