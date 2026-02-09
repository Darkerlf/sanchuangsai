// Neo4j LOAD CSV 示例（把 kg_export/*.csv 放到 Neo4j 的 import 目录后执行）
//
// 注意：
// 1) Neo4j Desktop：把 CSV 拖到 DB 的 Import 目录，或在设置中配置 import 路径
// 2) 服务器版：确保 neo4j.conf 中允许 file:// 访问

// ========== Nodes ==========
LOAD CSV WITH HEADERS FROM 'file:///node_brand.csv' AS row
MERGE (b:Brand {id: row.brand_id})
SET b.display = row.brand_display;

LOAD CSV WITH HEADERS FROM 'file:///node_product.csv' AS row
MERGE (p:Product {id: row.product_id})
SET p.title = row.title,
    p.price_num = toFloat(row.price_num),
    p.discount_rate = toFloat(row.discount_rate),
    p.rating = toFloat(row.product_rating),
    p.rating_count = toInteger(row.product_rating_count),
    p.bsr_category = row.bsr_category_norm,
    p.best_subcat = row.best_subcat_name,
    p.best_subcat_rank = toFloat(row.best_subcat_rank),
    p.bsr_rank = toFloat(row.bsr_rank),
    p.is_fba = toInteger(row.is_fba),
    p.has_aplus = toInteger(row.has_aplus),
    p.image_count = toInteger(row.image_count),
    p.bullet_count = toInteger(row.bullet_count),
    p.sales_proxy = toFloat(row.bought_count_number_clean);

LOAD CSV WITH HEADERS FROM 'file:///node_material.csv' AS row
MERGE (m:Material {id: row.material_id})
SET m.display_cn = row.material_display_cn;

LOAD CSV WITH HEADERS FROM 'file:///node_knife_type.csv' AS row
MERGE (t:KnifeType {id: row.knife_type_id})
SET t.display_cn = row.knife_type_display_cn;

LOAD CSV WITH HEADERS FROM 'file:///node_painpoint.csv' AS row
MERGE (pp:PainPoint {id: row.painpoint_id})
SET pp.display_cn = row.painpoint_display_cn;

// ========== Edges ==========
LOAD CSV WITH HEADERS FROM 'file:///edge_brand_sells_product.csv' AS row
MATCH (b:Brand {id: row.brand_id})
MATCH (p:Product {id: row.product_id})
MERGE (b)-[:SELLS]->(p);

LOAD CSV WITH HEADERS FROM 'file:///edge_product_has_material.csv' AS row
MATCH (p:Product {id: row.product_id})
MATCH (m:Material {id: row.material_id})
MERGE (p)-[r:HAS_MATERIAL]->(m)
SET r.source = row.source, r.confidence = toFloat(row.confidence);

LOAD CSV WITH HEADERS FROM 'file:///edge_product_has_knife_type.csv' AS row
MATCH (p:Product {id: row.product_id})
MATCH (t:KnifeType {id: row.knife_type_id})
MERGE (p)-[r:HAS_KNIFE_TYPE]->(t)
SET r.source = row.source, r.confidence = toFloat(row.confidence);

LOAD CSV WITH HEADERS FROM 'file:///edge_product_has_painpoint.csv' AS row
MATCH (p:Product {id: row.product_id})
MATCH (pp:PainPoint {id: row.painpoint_id})
MERGE (p)-[r:HAS_PAINPOINT]->(pp)
SET r.mention_n = toInteger(row.mention_n),
    r.neg_ratio = toFloat(row.neg_ratio),
    r.pos_ratio = toFloat(row.pos_ratio),
    r.avg_score = toFloat(row.avg_score);
