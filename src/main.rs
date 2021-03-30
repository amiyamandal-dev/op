mod engine;

fn liner_regression_process(){
    let mut train = engine::liner_regression::new("train.csv".to_string());
    let mut test = engine::liner_regression::new("test.csv".to_string());
    println!("liner regression!");
    let (b,m) = train.gradient_descent(0.0, 0.0, 0.0001, 10000);
    println!("b:{} m:{}",b, m);
    
    train.compute_error(b, m);

    test.predict(b, m);
}

fn main() {
    
}
