use wtest::Myllm;

fn main() {
    let mut myllm = Myllm::new();
    let res = myllm.infer("The sun is a deadly ");
    println!("response: {}", res);
}
