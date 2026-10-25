-- Warehouse layout (database "warehouse"). Applied idempotently by every load.
CREATE SCHEMA IF NOT EXISTS dwh;
CREATE SCHEMA IF NOT EXISTS features;

-- Clean layer: one row per oblast alert episode (raion / hromada records merged).
CREATE TABLE IF NOT EXISTS dwh.alerts (
    batch_month   text             NOT NULL,
    oblast        text             NOT NULL,
    started_at    timestamptz      NOT NULL,
    finished_at   timestamptz      NOT NULL,
    duration_min  double precision NOT NULL,
    n_records     integer          NOT NULL,
    end_is_guess  boolean          NOT NULL,
    source        text             NOT NULL,
    PRIMARY KEY (oblast, started_at)
);
CREATE INDEX IF NOT EXISTS alerts_batch_month_idx ON dwh.alerts (batch_month);
CREATE INDEX IF NOT EXISTS alerts_started_at_idx ON dwh.alerts (started_at);

-- One row per batch: where it came from, what each cleaning rule did, gate result.
CREATE TABLE IF NOT EXISTS dwh.batch_audit (
    batch_month     text PRIMARY KEY,
    source          text,
    raw_key         text,
    raw_sha256      text,
    rows_raw        integer,
    rows_duplicate  integer,
    rows_unfinished integer,
    rows_bad_time   integer,
    rows_valid      integer,
    rows_merged     integer,
    rows_clean      integer,
    invalid_share   double precision,
    m1_rows         integer,
    m2_rows         integer,
    gate_passed     boolean,
    gate_report     jsonb,
    dag_run_id      text,
    updated_at      timestamptz NOT NULL DEFAULT now()  -- audit metadata only, never a batch key
);

-- Feature layer. Labels sit next to features for training; serving ignores them.
CREATE TABLE IF NOT EXISTS features.m1_hourly (
    batch_month                  text             NOT NULL,
    hour_start                   timestamptz      NOT NULL,
    issue_time                   timestamptz      NOT NULL,
    hour_local                   integer          NOT NULL,
    lead                         integer          NOT NULL,
    issue_slot                   integer          NOT NULL,
    weekday                      integer          NOT NULL,
    kyiv_rate_same_hour_30d      double precision NOT NULL,
    kyiv_alert_min_prev_12h      double precision NOT NULL,
    days_since_long_alert        double precision NOT NULL,
    kyiv_under_alert_at_issue    integer          NOT NULL,
    neighbours_under_alert_at_issue integer          NOT NULL,
    neighbour_starts_prev_12h       integer          NOT NULL,
    label_start                  integer          NOT NULL,
    feature_version              text             NOT NULL,
    PRIMARY KEY (hour_start, feature_version)
);
CREATE INDEX IF NOT EXISTS m1_batch_month_idx ON features.m1_hourly (batch_month);

CREATE TABLE IF NOT EXISTS features.m2_alert (
    batch_month                  text             NOT NULL,
    alert_start                  timestamptz      NOT NULL,
    alert_end                    timestamptz      NOT NULL,
    start_hour_local             integer          NOT NULL,
    is_night                     integer          NOT NULL,
    weekday                      integer          NOT NULL,
    neighbours_under_alert_at_start integer          NOT NULL,
    neighbour_starts_prev_60m       integer          NOT NULL,
    oblasts_started_prev_5m         integer          NOT NULL,
    min_since_kyivska_start      double precision NOT NULL,
    min_since_prev_kyiv_end      double precision NOT NULL,
    prev_kyiv_duration_min       double precision NOT NULL,
    kyiv_alerts_prev_7d          integer          NOT NULL,
    duration_min                 double precision NOT NULL,
    label_valid                  boolean          NOT NULL,
    feature_version              text             NOT NULL,
    PRIMARY KEY (alert_start, feature_version)
);
CREATE INDEX IF NOT EXISTS m2_batch_month_idx ON features.m2_alert (batch_month);
