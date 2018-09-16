

fn longest_str_ref<'a>(x: &'a str, y: &'a str) -> &'a str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

/* The function below doesn't work because str is dynamic type where Rust compiler cannot
tell the size at compilation time. So use &str.
fn longest_str<'a>(x: str, y: str) -> str {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}
        println!("The longest str is {}", longest_str(string1, string2));

 */

fn longest_string<'a>(x: &'a String, y: &'a String) -> &'a String {
    if x.len() > y.len() {
        x
    } else {
        y
    }
}

fn main() {
    let string1 = String::from("long string is long");
    {
        let string2 = String::from("xyz");
        let result1 = longest_str_ref((&string1).as_str(), (&string2).as_str());
        let result2 = longest_string(&string1, &string2);

        println!("The longest str ref is {}", result1);
        println!("The longest string is {}", result2);
    }
    println!("Hello, world!");
}
