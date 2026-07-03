use common::ReconstructionResult;
use ndarray::{Array1, Array2};
use std::error::Error;
use chrono::Utc;
use uuid::Uuid;


/// Lê uma matriz de um arquivo CSV e infere suas dimensões pela estrutura do arquivo.
pub fn read_h_matrix_from_csv(file_path: &str) -> Result<Array2<f64>, Box<dyn Error>> {
    println!("[Servidor] Lendo matriz 'H' de: {}", file_path);
    let mut reader = csv::ReaderBuilder::new().has_headers(false).from_path(file_path)?;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for result in reader.records() {
        let record = result?;
        let row: Result<Vec<f64>, _> = record.iter().map(|field| field.trim().parse::<f64>()).collect();
        rows.push(row?);
    }
    if rows.is_empty() {
        return Err(format!("Erro de dimensão: O arquivo {} está vazio.", file_path).into());
    }
    let cols = rows[0].len();
    if cols == 0 {
        return Err(format!("Erro de dimensão: O arquivo {} tem uma linha vazia.", file_path).into());
    }
    if !rows.iter().all(|row| row.len() == cols) {
        return Err(format!("Erro de dimensão: O arquivo {} possui linhas de tamanhos diferentes.", file_path).into());
    }
    let flat_data: Vec<f64> = rows.into_iter().flatten().collect();
    Array2::from_shape_vec((flat_data.len() / cols, cols), flat_data).map_err(|e| e.into())
}

/// Lê um vetor de um arquivo CSV.
pub fn read_vector_from_csv(file_path: &str) -> Result<Array1<f64>, Box<dyn Error>> {
    println!("[Servidor] Lendo vetor de sinal de: {}", file_path);
    let mut reader = csv::ReaderBuilder::new().has_headers(false).from_path(file_path)?;
    let mut data = Vec::new();
    for result in reader.records() {
        let record = result?;
        let value: f64 = record.get(0)
            .ok_or_else(|| format!("Linha vazia no arquivo {}.", file_path))?
            .trim()
            .parse()?;
        data.push(value);
    }
    Ok(Array1::from(data))
}


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
        let gamma_l = (100.0 + 0.05 * (l as f64) * (l as f64).sqrt());
        g[l] *= gamma_l;
    }
    let mut f = Array1::<f64>::zeros(n);
    let mut r = g;
    let z = h.t().dot(&r);
    let mut p = z.clone();
    let mut z_t_z = z.dot(&z);
    let max_iterations = 10;
    let mut iterations = 0;
    let convergence_threshold = 1e-4;

    for iteration_count in 0..max_iterations {
        iterations = iteration_count + 1;
        let w = h.dot(&p);
        let w_t_w = w.dot(&w);
        if w_t_w.abs() < 1e-20 { break; }
        let alpha = z_t_z / w_t_w;
        f += &(&p * alpha);
        r -= &(&w * alpha);
        let z_next = h.t().dot(&r);
        let z_t_z_next = z_next.dot(&z_next);
        if z_t_z_next < convergence_threshold {
            println!("[Servidor] CGNR convergiu na iteração {}", iterations);
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
        image_pixels, // 
        iterations,
        f,
    }
}


fn spectral_norm_hth(h: &Array2<f64>, power_iterations: usize) -> f64 {
    let n = h.shape()[1];
    let mut v = Array1::<f64>::from_elem(n, 1.0 / (n as f64).sqrt());

    for _ in 0..power_iterations {
        let hv = h.dot(&v);
        let htv = h.t().dot(&hv);
        let norm = htv.dot(&htv).sqrt();
        if norm < 1e-20 {
            break;
        }
        v = &htv / norm;
    }

    let hv = h.dot(&v);
    let htv = h.t().dot(&hv);
    let v_dot_v = v.dot(&v);
    if v_dot_v.abs() < 1e-20 {
        return 0.0;
    }
    v.dot(&htv) / v_dot_v
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

    // --- Ganho de sinal (γ_l), l = 1..S ---
    let mut g = g_signal.clone();
    for l in 0..s {
        let l_1idx = (l + 1) as f64;
        let gamma_l = 100.0 + 0.05 * l_1idx * l_1idx.sqrt();
        g[l] *= gamma_l;
    }

    // --- Fator de redução: c = ||H^T * H||_2 ---
    let c = spectral_norm_hth(h, 100);
    let c_safe = if c.abs() < 1e-20 { 1.0 } else { c };
    println!("[CGNE] fator de redução c = {:.6}", c_safe);

    // --- H^T * g (usado tanto para λ quanto para o resíduo inicial) ---
    let htg = h.t().dot(&g);

    // --- Coeficiente de regularização: λ = max(abs(H^T*g)) * 0.10 ---
    let lambda = htg.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs())) * 0.10;
    println!("[CGNE] lambda = {:.6}", lambda);

    let lambda_scaled = lambda / c_safe;

    // Aplica M = (H^T H + λI) / c a um vetor p, sem montar H^T H explicitamente
    let apply_m = |p: &Array1<f64>| -> Array1<f64> {
        let hp = h.dot(p);
        let hthp = h.t().dot(&hp);
        (&hthp / c_safe) + &(p * lambda_scaled)
    };

    // --- Inicialização (f0 = 0 => r0 = H^T*g / c, resolvendo M*f = H^T*g / c) ---
    let mut f = Array1::<f64>::zeros(n);
    let mut r = &htg / c_safe;
    let mut r_norm = r.dot(&r).sqrt();
    let mut p = r.clone();
    let mut r_t_r = r.dot(&r);

    let max_iterations = 10;
    let mut iterations = 0;
    let convergence_threshold = 1e-4;

    println!("[CGNE] ||r_0|| = {:.6}", r_norm);

    for iteration_count in 0..max_iterations {
        iterations = iteration_count + 1;

        let mp = apply_m(&p);
        let p_t_mp = p.dot(&mp);
        if p_t_mp.abs() < 1e-20 {
            println!("[CGNE] p^T*M*p ~ 0, parando (direção degenerada)");
            break;
        }

        let alpha = r_t_r / p_t_mp;
        f += &(&p * alpha);
        r -= &(&mp * alpha);

        let r_norm_next = r.dot(&r).sqrt();
        let epsilon = (r_norm_next - r_norm).abs();
        let r_t_r_next = r.dot(&r);

        println!(
            "[CGNE] iter {}: ||r|| = {:.6}, epsilon = {:.6}",
            iterations, r_norm_next, epsilon
        );

        if epsilon < convergence_threshold {
            println!("[Servidor] CGNE convergiu na iteração {}", iterations);
            break;
        }

        if r_t_r.abs() < 1e-20 {
            println!("[CGNE] r^T*r ~ 0, parando (resíduo nulo)");
            break;
        }

        let beta = r_t_r_next / r_t_r;
        p = &r + &(&p * beta);

        r_t_r = r_t_r_next;
        r_norm = r_norm_next;
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
        iterations,
        f,
    }
}

/// Salva a imagem reconstruída e retorna o nome do arquivo.
pub fn save_image(result: &ReconstructionResult) -> Result<String, Box<dyn Error>> {
    let (height, width) = result.image_pixels;
    if height * width != result.f.len() {
        return Err("Dimensões da imagem não correspondem ao tamanho do vetor 'f'".into());
    }

    let f_slice = result.f.as_slice().unwrap();
    let mut image_values = Vec::with_capacity(result.f.len());

    // Reordena os valores no mesmo formato Fortran do Python (reshape(order='F'))
    // e inverte verticalmente para corresponder ao comportamento anterior.
    for row in (0..height).rev() {
        for col in 0..width {
            image_values.push(f_slice[col * height + row].abs());
        }
    }

    let f_min = image_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let f_max = image_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let range = f_max - f_min;

    let image_buffer: Vec<u8> = if range.abs() < 1e-9 {
        vec![0; image_values.len()]
    } else {
        image_values.iter().map(|&val| (((val - f_min) / range) * 255.0) as u8).collect()
    };
    
    let file_name = format!("img_{}_{}_{}.png", result.user_id, result.algorithm_id, result.end_time.timestamp());

    image::save_buffer(&file_name, &image_buffer, width as u32, height as u32, image::ColorType::L8)?;
    
    println!("[Servidor] Imagem invertida salva como: {}", file_name);

    Ok(file_name)
}
