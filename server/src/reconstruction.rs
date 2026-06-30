use common::ReconstructionResult;
use ndarray::{Array1, Array2};
use std::error::Error;
use chrono::Utc;
use uuid::Uuid;


/// Lê uma matriz de um arquivo CSV.
pub fn read_h_matrix_from_csv(file_path: &str, s_samples: usize, n_pixels: usize) -> Result<Array2<f64>, Box<dyn Error>> {
    println!("[Servidor] Lendo matriz 'H' de: {}", file_path);
    let mut reader = csv::ReaderBuilder::new().has_headers(false).from_path(file_path)?;
    let mut flat_data = Vec::with_capacity(s_samples * n_pixels);
    for result in reader.records() {
        let record = result?;
        for field in record.iter() {
            let value: f64 = field.trim().parse()?;
            flat_data.push(value);
        }
    }
    if flat_data.len() != s_samples * n_pixels {
        return Err(format!("Erro de dimensão: O arquivo {} contém {} elementos, mas eram esperados {}.", 
            file_path, flat_data.len(), s_samples * n_pixels).into());
    }
    Array2::from_shape_vec((s_samples, n_pixels), flat_data).map_err(|e| e.into())
}

// ALTERADO: A assinatura da função agora aceita `image_pixels`.
pub fn execute_cgnr(
    algorithm_id: &str,
    user_id: Uuid,
    h: &Array2<f64>,
    g_signal: &Array1<f64>,
    image_pixels: (usize, usize),
) -> ReconstructionResult {
    let start_time = Utc::now();
    let s = h.shape()[0];
    let n = h.shape()[1];
    
    // A lógica do algoritmo permanece a mesma
    let mut g = g_signal.clone();
    for l in 0..s {
        let l_f = l as f64;
        let gamma_l = 100.0 + 0.05 * l_f * l_f.sqrt();
        g[l] *= gamma_l;
    }
    let mut f = Array1::<f64>::zeros(n);
    let mut r = g;
    let z = h.t().dot(&r);
    let mut p = z.clone();
    let mut z_t_z = z.dot(&z);
    let max_iterations = 10;
    let mut i = 0;
    let convergence_threshold = 1e-4;

    for iteration_count in 0..max_iterations {
        i = iteration_count;
        let w = h.dot(&p);
        let w_t_w = w.dot(&w);
        if w_t_w.abs() < 1e-20 { break; }
        let alpha = z_t_z / w_t_w;
        f += &(&p * alpha);
        r -= &(&w * alpha);
        let z_next = h.t().dot(&r);
        let z_t_z_next = z_next.dot(&z_next);
        if z_t_z_next < convergence_threshold {
            println!("[Servidor] CGNR convergiu na iteração {}", i + 1);
            break;
        }
        if z_t_z.abs() < 1e-20 { break; }
        let beta = z_t_z_next / z_t_z;
        p = &z_next + &(&p * beta);
        z_t_z = z_t_z_next;
    }

    let end_time = Utc::now();
    let reconstruction_time_ms = end_time.signed_duration_since(start_time).num_milliseconds();

    ReconstructionResult {
        user_id,
        algorithm_id: algorithm_id.to_string(),
        start_time,
        end_time,
        reconstruction_time_ms,
        image_pixels,
        iterations: i + 1,
        f,
    }
}


pub fn execute_cgne(
    algorithm_id: &str,
    user_id: Uuid,
    h: &Array2<f64>,
    g_signal: &Array1<f64>,
    image_pixels: (usize, usize),
) -> ReconstructionResult {
    let start_time = Utc::now();
    let s = h.shape()[0];
    let n = h.shape()[1];

    let mut g = g_signal.clone();
    for l in 0..s {
        let l_f = l as f64;
        let gamma_l = 100.0 + 0.05 * l_f * l_f.sqrt();
        g[l] *= gamma_l;
    }

    let mut f = Array1::<f64>::zeros(n);
    let mut r = g.clone();
    let mut p = h.t().dot(&r);
    let mut r_t_r = r.dot(&r);

    let max_iterations = 3;//
    let mut i = 0;
    let convergence_threshold = 1e-4;

    for iteration_count in 0..max_iterations {
        i = iteration_count;
        let p_t_p = p.dot(&p);
        if p_t_p.abs() < 1e-20 {
            break;
        }
        let w = h.dot(&p);
        let alpha = r_t_r / p_t_p;
        f += &(&p * alpha);
        r -= &(&w * alpha);

        let r_t_r_next = r.dot(&r);
        if r_t_r_next < convergence_threshold {
            println!("[Servidor] CGNE convergiu na iteração {}", i + 1);
            break;
        }

        if r_t_r.abs() < 1e-20 {
            break;
        }
        let beta = r_t_r_next / r_t_r;
        p = h.t().dot(&r) + &(&p * beta);
        r_t_r = r_t_r_next;
    }

    let end_time = Utc::now();
    let reconstruction_time_ms = end_time.signed_duration_since(start_time).num_milliseconds();

    ReconstructionResult {
        user_id,
        algorithm_id: algorithm_id.to_string(),
        start_time,
        end_time,
        reconstruction_time_ms,
        image_pixels,
        iterations: i + 1,
        f,
    }
}


/// Salva a imagem reconstruída e retorna o nome do arquivo.
pub fn save_image(result: &ReconstructionResult) -> Result<String, Box<dyn Error>> {
    let (height, width) = result.image_pixels;
    if height * width != result.f.len() {
        return Err("Dimensões da imagem não correspondem ao tamanho do vetor 'f'".into());
    }

    // Inverte a imagem verticalmente para a orientação correta
    let mut flipped_f = Vec::with_capacity(result.f.len());
    for row_chunk in result.f.as_slice().unwrap().chunks(width).rev() {
        flipped_f.extend_from_slice(row_chunk);
    }
    
    // Normaliza os valores de f64 para u8 (0-255)
    let f_min = flipped_f.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let f_max = flipped_f.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = f_max - f_min;

    let image_buffer: Vec<u8> = if range.abs() < 1e-9 {
        vec![0; flipped_f.len()]
    } else {
        flipped_f.iter().map(|&val| (((val - f_min) / range) * 255.0) as u8).collect()
    };
    
    let file_name = format!("img_{}_{}_{}.png", result.user_id, result.algorithm_id, result.end_time.timestamp());

    image::save_buffer(&file_name, &image_buffer, width as u32, height as u32, image::ColorType::L8)?;
    
    println!("[Servidor] Imagem invertida salva como: {}", file_name);

    Ok(file_name)
}