extern crate num;
use num::Complex;
use std::str::FromStr;

#[allow(dead_code)]
fn complex_square_add_loop(c: Complex<f64>) {
    let mut z = Complex { re: 0.0, im: 0.0 };
    loop {
        z = z * z + c;
    }
}

fn escape_time(c : Complex<f64>, limit: u32) -> Option<u32> {
    let mut z = Complex{ re: 0.0, im: 0.0 };
    for i in 0..limit{
        z = z * z + c;
        if z.norm_sqr() > 4.0 {
            return Some(i);
        }
    }
    None
}

fn parse_pair<T: FromStr>(s: &str, separator: char) -> Option<(T, T)> {
    match s.find(separator) {
        None => None,
        Some(index) => {
            match(T::from_str(&s[..index]), T::from_str(&s[index + 1..])) {
                (Ok(l), Ok(r)) => Some((l, r)),
                _ => None
            }
        }
    }
}

#[test]
fn test_parse_pair() {
    // i32 and f64 implement FromStr trait
    assert_eq!(parse_pair::<i32>("", ','), None);
    assert_eq!(parse_pair::<f64>("0.55x1.53", 'x'), Some((0.55, 1.53)));
}

fn parse_complex(s: &str) -> Option<Complex<f64>> {
    match parse_pair(s, ',') {
        Some((re, im)) => Some(Complex { re, im }),
        None => None
    }
}
#[test]
fn test_parse_complex() {
    assert_eq!(parse_complex("6.7,9.1"), Some(Complex{ re: 6.7, im: 9.1}));
    assert_eq!(parse_complex("6.7,9,1"), None);
}

fn pixel_to_point(bounds: (usize, usize),
pixel: (usize, usize),
upper_left: Complex<f64>,
lower_right: Complex<f64>) -> Complex<f64>{
    let (width, height) = (lower_right.re - upper_left.re, upper_left.im - lower_right.im);

    Complex {
        re: upper_left.re + pixel.0 as f64 * width / bounds.0 as f64,
        im: upper_left.im - pixel.1 as f64 * height / bounds.1 as f64
    }
}

fn render(pixels: &mut [u8],
    bounds: (usize, usize),
    upper_left: Complex<f64>,
    lower_right: Complex<f64>) {
    assert!(pixels.len() == bounds.0 * bounds.1);

    for row in 0 .. bounds.1 {
        for column in 0 .. bounds.0 {
            let point = pixel_to_point(bounds, (column, row), upper_left, lower_right);
            pixels[row * bounds.0 + column] = 
                match escape_time(point, 255) {
                    None => 0,
                    Some(count) => 255 - count as u8
                };
            
        }
    }
}

extern crate image;
use image::ColorType;
use image::png::PNGEncoder;
use std::fs::File;

fn write_image(filename: &str, pixels: &[u8], bounds: (usize, usize))-> Result<(), std::io::Error> {
    let output = File::create(filename)?;
    let encoder = PNGEncoder::new(output);
    encoder.encode(&pixels,
        bounds.0 as u32, bounds.1 as u32,
            ColorType::Gray(8))?;
    Ok(())
}

// To avoid: no method named `write_fmt` found for type `std::io::Stderr` 
use std::io::Write;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        writeln!(std::io::stderr(),
            "Example: {} mandel.png 1000x750 -1.20,0.35 -1,0.20", args[0]).unwrap();
        std::process::exit(1);
    }

    // Without parse pair, it raises: no field `0` on type `std::option::Option<(_, _)>`
    let bounds = parse_pair(&args[2], 'x').expect("err parse pair");
    let upper_left = parse_complex(&args[3]).expect("err upper_left");
    let lower_right = parse_complex(&args[4]).expect("err lower_right");
    let mut pixels = vec![0; bounds.0 * bounds.1];
    render(&mut pixels, bounds, upper_left, lower_right);
    write_image(&args[1], &pixels, bounds).expect("err png file");

}
