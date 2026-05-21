-- ============================================================
--  JACOB AI | AgriSense Pro — MySQL Database Schema & Queries
--  Database: agrisense_db
-- ============================================================

CREATE DATABASE IF NOT EXISTS agrisense_db
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE agrisense_db;

-- ─────────────────────────────────────────────────────────────
-- TABLE 1: crops — master crop reference
-- ─────────────────────────────────────────────────────────────
CREATE TABLE crops (
    crop_id      INT AUTO_INCREMENT PRIMARY KEY,
    crop_name    VARCHAR(100) NOT NULL UNIQUE,
    category     ENUM('Cereals','Pulses','Oilseeds','Cash Crops','Other')
                 NOT NULL DEFAULT 'Other',
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ─────────────────────────────────────────────────────────────
-- TABLE 2: seasons — Kharif / Rabi / Summer
-- ─────────────────────────────────────────────────────────────
CREATE TABLE seasons (
    season_id    INT AUTO_INCREMENT PRIMARY KEY,
    season_name  ENUM('Kharif','Rabi','Summer','Total') NOT NULL UNIQUE
);

-- ─────────────────────────────────────────────────────────────
-- TABLE 3: crop_observations — core fact table
-- ─────────────────────────────────────────────────────────────
CREATE TABLE crop_observations (
    obs_id         BIGINT AUTO_INCREMENT PRIMARY KEY,
    crop_id        INT NOT NULL,
    season_id      INT NOT NULL,
    year_label     VARCHAR(10) NOT NULL,   -- e.g. '2022-23'
    year_start     SMALLINT NOT NULL,      -- e.g. 2022
    area_lakh_ha   FLOAT,                  -- Lakh Hectares
    production_mt  FLOAT,                  -- Million Tonnes
    yield_kg_ha    FLOAT,                  -- Kg per Hectare
    data_source    VARCHAR(50) DEFAULT 'DAC&FW India',
    inserted_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (crop_id)    REFERENCES crops(crop_id),
    FOREIGN KEY (season_id)  REFERENCES seasons(season_id),
    INDEX idx_crop_year   (crop_id, year_start),
    INDEX idx_season_year (season_id, year_start)
);

-- ─────────────────────────────────────────────────────────────
-- TABLE 4: ml_predictions — store model outputs
-- ─────────────────────────────────────────────────────────────
CREATE TABLE ml_predictions (
    pred_id        BIGINT AUTO_INCREMENT PRIMARY KEY,
    obs_id         BIGINT,
    model_name     VARCHAR(60) NOT NULL,
    predicted_yield FLOAT NOT NULL,
    actual_yield    FLOAT,
    abs_error       FLOAT GENERATED ALWAYS AS
                    (ABS(predicted_yield - actual_yield)) STORED,
    predicted_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (obs_id) REFERENCES crop_observations(obs_id)
);

-- ─────────────────────────────────────────────────────────────
-- SEED: seasons
-- ─────────────────────────────────────────────────────────────
INSERT IGNORE INTO seasons (season_name)
VALUES ('Kharif'), ('Rabi'), ('Summer'), ('Total');

-- ─────────────────────────────────────────────────────────────
-- SEED: crops with categories
-- ─────────────────────────────────────────────────────────────
INSERT IGNORE INTO crops (crop_name, category) VALUES
('Rice',                    'Cereals'),
('Wheat',                   'Cereals'),
('Maize',                   'Cereals'),
('Barley',                  'Cereals'),
('Jowar',                   'Cereals'),
('Bajra',                   'Cereals'),
('Ragi',                    'Cereals'),
('Small Millets',           'Cereals'),
('Tur',                     'Pulses'),
('Gram',                    'Pulses'),
('Urad',                    'Pulses'),
('Moong',                   'Pulses'),
('Lentil',                  'Pulses'),
('Other Pulses',            'Pulses'),
('Groundnut',               'Oilseeds'),
('Castorseed',              'Oilseeds'),
('Sesamum',                 'Oilseeds'),
('Soybean',                 'Oilseeds'),
('Sunflower',               'Oilseeds'),
('Rapeseed & Mustard',      'Oilseeds'),
('Linseed',                 'Oilseeds'),
('Safflower',               'Oilseeds'),
('Sugarcane',               'Cash Crops'),
('Cotton',                  'Cash Crops'),
('Jute',                    'Cash Crops'),
('Tobacco',                 'Cash Crops');

-- ─────────────────────────────────────────────────────────────
-- SAMPLE INSERT (one observation row — replicate for all rows)
-- In production, use Python's mysql-connector to bulk insert
-- ─────────────────────────────────────────────────────────────
INSERT INTO crop_observations
    (crop_id, season_id, year_label, year_start, area_lakh_ha, production_mt, yield_kg_ha)
SELECT
    c.crop_id,
    s.season_id,
    '2022-23',
    2022,
    410.38,  -- area in lakh ha
    1115.09, -- production in lakh tonnes (source value)
    2780.0   -- yield Kg/Ha
FROM crops c, seasons s
WHERE c.crop_name = 'Rice' AND s.season_name = 'Kharif';

-- ─────────────────────────────────────────────────────────────
-- ANALYTICAL QUERIES
-- ─────────────────────────────────────────────────────────────

-- Q1: Average yield per crop (all years, Kharif season only)
SELECT
    c.crop_name,
    ROUND(AVG(o.yield_kg_ha), 2)    AS avg_yield_kg_ha,
    ROUND(AVG(o.area_lakh_ha), 2)   AS avg_area_lakh_ha,
    ROUND(AVG(o.production_mt), 2)  AS avg_production_mt
FROM crop_observations o
JOIN crops c   ON c.crop_id    = o.crop_id
JOIN seasons s ON s.season_id  = o.season_id
WHERE s.season_name = 'Kharif'
GROUP BY c.crop_name
ORDER BY avg_yield_kg_ha DESC;

-- Q2: Year-on-year yield trend for Rice
SELECT
    year_label,
    year_start,
    season_name,
    yield_kg_ha,
    ROUND(yield_kg_ha - LAG(yield_kg_ha) OVER
          (PARTITION BY crop_name, season_name ORDER BY year_start), 2)
          AS yoy_change_kg_ha
FROM crop_observations o
JOIN crops   c ON c.crop_id   = o.crop_id
JOIN seasons s ON s.season_id = o.season_id
WHERE c.crop_name = 'Rice'
ORDER BY season_name, year_start;

-- Q3: Crops with highest production growth (2021 → 2024)
SELECT
    c.crop_name,
    MIN(CASE WHEN o.year_start = 2021 THEN o.production_mt END) AS prod_2021,
    MAX(CASE WHEN o.year_start = 2024 THEN o.production_mt END) AS prod_2024,
    ROUND(
        (MAX(CASE WHEN o.year_start = 2024 THEN o.production_mt END) -
         MIN(CASE WHEN o.year_start = 2021 THEN o.production_mt END))
        / NULLIF(MIN(CASE WHEN o.year_start = 2021 THEN o.production_mt END), 0) * 100,
    2) AS growth_pct
FROM crop_observations o
JOIN crops c ON c.crop_id = o.crop_id
JOIN seasons s ON s.season_id = o.season_id
WHERE s.season_name = 'Total' AND o.year_start IN (2021, 2024)
GROUP BY c.crop_name
HAVING prod_2021 IS NOT NULL AND prod_2024 IS NOT NULL
ORDER BY growth_pct DESC;

-- Q4: Category-level production summary
SELECT
    c.category,
    SUM(o.production_mt)            AS total_production_mt,
    ROUND(AVG(o.yield_kg_ha), 1)    AS avg_yield_kg_ha,
    COUNT(DISTINCT c.crop_id)       AS crop_count
FROM crop_observations o
JOIN crops c ON c.crop_id = o.crop_id
JOIN seasons s ON s.season_id = o.season_id
WHERE s.season_name = 'Total'
GROUP BY c.category
ORDER BY total_production_mt DESC;

-- Q5: Yield anomaly detection (>2 SD from crop mean)
WITH stats AS (
    SELECT
        crop_id,
        AVG(yield_kg_ha)    AS mean_yield,
        STDDEV(yield_kg_ha) AS std_yield
    FROM crop_observations
    WHERE yield_kg_ha IS NOT NULL
    GROUP BY crop_id
)
SELECT
    c.crop_name,
    s.season_name,
    o.year_label,
    o.yield_kg_ha,
    ROUND(st.mean_yield, 1) AS crop_mean,
    ROUND(ABS(o.yield_kg_ha - st.mean_yield) / st.std_yield, 2) AS z_score
FROM crop_observations o
JOIN crops   c  ON c.crop_id   = o.crop_id
JOIN seasons s  ON s.season_id = o.season_id
JOIN stats   st ON st.crop_id  = o.crop_id
WHERE ABS(o.yield_kg_ha - st.mean_yield) > 2 * st.std_yield
ORDER BY z_score DESC;

-- Q6: Model performance stored in ml_predictions
SELECT
    model_name,
    COUNT(*)                                AS predictions,
    ROUND(AVG(abs_error), 2)                AS avg_absolute_error,
    ROUND(SQRT(AVG(POW(predicted_yield - actual_yield, 2))), 2) AS rmse
FROM ml_predictions
GROUP BY model_name
ORDER BY rmse ASC;
