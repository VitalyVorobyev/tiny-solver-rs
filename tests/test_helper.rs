#[cfg(test)]
mod tests {
    use tiny_solver::helper::read_g2o;

    #[test]
    fn read_g2o_tolerates_whitespace_and_unknown_lines() {
        // Tabs, repeated spaces, a blank line and a tag this reader does not
        // handle (FIX) must not stop parsing of the lines after them.
        let content = "VERTEX_SE2 0 0.0 0.0 0.0\n\
                       VERTEX_SE2\t1\t1.0\t0.0\t0.0\n\
                       \n\
                       FIX 0\n\
                       VERTEX_SE2  2  2.0  0.0  0.0\n\
                       EDGE_SE2 0 1 1.0 0.0 0.0 1 0 0 1 0 1\n\
                       EDGE_SE2\t1\t2\t1.0\t0.0\t0.0\t1\t0\t0\t1\t0\t1\n";
        let path =
            std::env::temp_dir().join(format!("tiny_solver_read_g2o_{}.g2o", std::process::id()));
        std::fs::write(&path, content).unwrap();

        let (problem, initial_values) = read_g2o(path.to_str().unwrap());
        std::fs::remove_file(&path).unwrap();

        assert_eq!(initial_values.len(), 3);
        assert_eq!(initial_values["x1"][1], 1.0);
        assert_eq!(initial_values["x2"][1], 2.0);
        // two 3-dim edges plus the 3-dim prior on x0
        assert_eq!(problem.total_residual_dimension, 9);
    }
}
