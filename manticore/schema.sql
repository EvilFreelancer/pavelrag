CREATE TABLE IF NOT EXISTS `nodes` (
    id bigint,
    document text indexed stored,
    embedding float_vector
        knn_type='hnsw'
        knn_dims='1024'
        hnsw_similarity='L2'
        hnsw_m='16'
        hnsw_ef_construction='100',
    metadata json,
    uuid text indexed stored
);

CREATE TABLE IF NOT EXISTS `edges` (
    from_id bigint,
    to_id bigint,
    weight int
);
