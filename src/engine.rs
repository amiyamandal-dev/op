use std::error::Error;

use ndarray::prelude::*;

#[derive(Debug, Clone)]
pub(crate) struct liner_regression {
    x: Vec<f64>,
    y:Vec<f64>
}

fn read_csv(file_path:String) ->Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    let mut rdr = match csv::Reader::from_path(file_path.clone()){
        Ok(r) => {r}
        Err(_) => {
            panic!("file dont exist {}", file_path);
        }
    };
    let mut x = vec![];
    let mut y = vec![];
    for r in rdr.records(){
        let record = match r{
            Ok(r) =>  r,
            Err(er)=>{
                panic!(er)
            }
        };
        x.push(record[0].parse::<f64>().unwrap());
        y.push(record[1].parse::<f64>().unwrap());
    }
    Ok((x, y))
}


impl liner_regression {
    pub fn new(path_file:String) -> Self {
        let (t_x, t_y) = read_csv(path_file).unwrap();
        liner_regression{
            x: t_x,
            y: t_y,
        }
    }

    fn step_gradient_descent(&mut self, m_current:f64, b_current:f64, learning_rate:f64)-> (f64, f64) {
        let mut b_gradient = 0.0;
        let mut m_gradient = 0.0;
        let N = self.x.len();
        for i in 0 .. N {
            b_gradient += -((2.0 / N as f64) * (self.y[i] - ((m_current * self.x[i]) + b_current)));
            m_gradient += -(2.0 / N as f64) * self.x[i] * (self.y[i] - ((m_current * self.x[i]) + b_current));
        }
        let new_b = b_current - (learning_rate*b_gradient);
        let new_m = m_current - (learning_rate *m_gradient);

        (new_b, new_m)
    }

    pub fn gradient_descent(&mut self, starting_b :f64, starting_m :f64, learing_rate :f64, num_iteartion:i64) ->(f64, f64){
        let mut b = starting_b;
        let mut m = starting_m;

        for i in 0..num_iteartion{
            let (t_b, t_m) = self.step_gradient_descent(m, b, learing_rate);
            b = t_b;
            m = t_m;
        }
        (b, m)
    }

    pub fn compute_error(&mut self, b:f64, m:f64){
        let mut total_error = 0.0;
        let N = self.x.len();

        for i in 0..N{
            let cal_val = self.y[i] - (m * self.x[i] +b);
            total_error += cal_val.exp2();
        }
        println!("error rate::>> {}", total_error/N as f64);

    }

    fn percentage(&self,predval :f64, exacval :f64){
        let errorrate = ((exacval - predval) * 100.0) / exacval;
        println!("error rate:: {} {} >> {}", predval, exacval, errorrate);

    }

    pub fn predict(&mut self, b:f64, m:f64){
        let N = self.x.len();

        for i in 0..N {
            let y_pred = m*self.x[i] + b;
            self.percentage(y_pred, self.y[i]);
        }
    }
}
