extern crate xml;
use std::env;
use std::fs::File;
use std::io::BufReader;

use xml::reader::{EventReader, XmlEvent};

fn indent(size: usize) -> String {
    const INDENT: &'static str = "  ";
    (0..size).map(|_| INDENT)
             .fold(String::with_capacity(size*INDENT.len()), |r, s| r + s)
}

fn main() {
    let args: Vec<String> = env::args().collect();
    println!("{:?}", args);
    let jawikitionary_filename = &args[1];
    let file = File::open(jawikitionary_filename).unwrap();
    let file = BufReader::new(file);

    let parser = EventReader::new(file);
    let mut depth = 0;
    let mut last_tag_name = String::from("");
    let mut title_count:u64 = 0;
    for e in parser {
        match e {
            Ok(XmlEvent::StartElement { name, attributes, .. }) => {
                //let attributes: Vec<xml::attribute::OwnedAttribute>  = attributes_;
                let local_name:String = name.local_name;
                last_tag_name = local_name;
                let _ : Vec<xml::attribute::OwnedAttribute> = attributes;
                // println!("{}+{}", indent(depth), local_name);
                depth += 1;
            }
            Ok(XmlEvent::Characters(c_) ) => {
                let c: String = c_;
                if last_tag_name == "title" {
                    println!("title: {}", indent(depth), c);
                    title_count += 1;
                }
            }
            Ok(XmlEvent::EndElement { name: _ }) => {
                depth -= 1;
                // println!("{}-{}", indent(depth), name);
            }
            Err(e) => {
                println!("Error: {}", e);
                break;
            }
            _ => {}
        }
    }
    println!("Title count: {}", title_count);
}
