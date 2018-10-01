extern crate csv;
use std::fs::File;
use std::time::Instant;
use ansi_term::Colour::*;
use ansi_term::{ANSIString, ANSIStrings, Style};
use std::result::*;

use ndarray::prelude::*;

pub type Grayscale = f32;

pub const IMG_H_SIZE: usize = 28;
pub const IMG_W_SIZE: usize = 28;
//const HIDDEN_LAYER_SIZE: usize = 100;

pub const MNIST_DOT_MAX: f32 = 255.;

pub struct MnistRecord {
    pub label: i32,
    pub dots: [Grayscale; IMG_H_SIZE * IMG_W_SIZE],
    pub dots_array: Array2<f32>,
}

impl MnistRecord {
    pub fn load_from_csv(file_path: &str) -> Result<Vec<MnistRecord>, String> {
        let before_record = Instant::now();

        let file = try!(File::open(file_path).map_err(|e| e.to_string()));
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(false)
            .from_reader(file);
        let mut mnist_records: Vec<MnistRecord> = Vec::new();
        for result in rdr.records() {
            let mut array: [Grayscale; IMG_W_SIZE * IMG_W_SIZE] = [0.; IMG_W_SIZE * IMG_W_SIZE];
            let record = result.unwrap();
            assert!(record.len() == 1 + 28 * 28);
            for i in 1..(IMG_W_SIZE * IMG_W_SIZE) {
                let v: f32 = record[i + 1].parse().unwrap();
                let vv: f32 = v / MNIST_DOT_MAX;
                array[i] = vv;
                debug_assert!(vv >= 0. && vv <= 1., "MNIST dot must be between 0 - 255");
            }
            let label: i32 = record[0].parse().unwrap();
            let vv: Vec<f32> = array.to_vec();
            //        let dots_array1: Array1<f32> = Array1::from_vec(vv);
            let dots_array2: Array2<f32> = Array2::from_shape_vec((1, 784), vv).unwrap();
            let mnist: MnistRecord = MnistRecord {
                label,
                dots: array,
                dots_array: dots_array2,
            };
            mnist_records.push(mnist);
        }
        println!("Read {} for {} MNIST records in {} secs",
            file_path,
            mnist_records.len(), 
            before_record.elapsed().as_secs()
        );
        Ok(mnist_records)
    }
    pub fn print(&self) {
        let mut s = String::new();
        let mut v: Vec<ANSIString<'static>> = Vec::new();
        for i in 0..IMG_H_SIZE {
            for j in 0..IMG_W_SIZE {
                if self.dots[i * IMG_H_SIZE + j] > 0. {
                    s.push_str("O");
                    let c = self.dots[i * IMG_H_SIZE + j];
                    debug_assert!(c >= 0. && c <= 1., "MNIST dot must be 0 - 255");
                    let p = (1. - c);
                    if p < 0. {
                        println!("p : {}", p);
                    }
                    let r = 255. - 232.;
                    // The bigger c is, the darker
                    let term_color = (255. - (r * c)) as u8;
                    debug_assert!(term_color >= 232, "xterm color range. u8 maximum is 255.");
                    let color = Fixed(term_color); // RGB(c256, c256, c256);
                    let t = Black.on(color).paint(" ");
                    v.push(t);
                } else {
                    s.push_str(" ");
                    let t = White.on(Fixed(231)).paint(" ");
                    v.push(t);
                }
            }
            s.push_str("\n");
            v.push(Style::default().paint("\n"));
        }
        // print!("Label {}:\n{}", self.label, s);
        print!("Image:\n{}", ANSIStrings(&v));
    }
}



