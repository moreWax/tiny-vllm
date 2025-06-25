use anyhow::Result;

/// Clamp `value` to the inclusive range [min_value, max_value].
pub fn clamp(value: i64, min_value: i64, max_value: i64) -> i64 {
    if value < min_value {
        min_value
    } else if value > max_value {
        max_value
    } else {
        value
    }
}

/// Flatten a vector of vectors into a single vector.
pub fn flatten(list_of_lists: Vec<Vec<i64>>) -> Vec<i64> {
    list_of_lists.into_iter().flatten().collect()
}

/// Split a vector into chunks of a given size.
pub fn chunked(lst: Vec<i64>, size: usize) -> Vec<Vec<i64>> {
    if size == 0 {
        return Vec::new();
    }
    lst.chunks(size).map(|c| c.to_vec()).collect()
}
