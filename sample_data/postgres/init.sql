-- Seeds the demo Postgres container with the same data as sample_data/out/*.csv.
CREATE TABLE orders (
    order_id     INTEGER PRIMARY KEY,
    order_date   DATE NOT NULL,
    customer_id  INTEGER NOT NULL,
    region       TEXT NOT NULL,
    category     TEXT NOT NULL,
    channel      TEXT NOT NULL,
    units        INTEGER NOT NULL,
    unit_price   NUMERIC(10, 2) NOT NULL,
    discount     NUMERIC(4, 2) NOT NULL,
    revenue      NUMERIC(12, 2) NOT NULL,
    promo        BOOLEAN NOT NULL
);

CREATE TABLE customers (
    customer_id      INTEGER PRIMARY KEY,
    signup_date      DATE NOT NULL,
    region           TEXT NOT NULL,
    plan             TEXT NOT NULL,
    tenure_months    INTEGER NOT NULL,
    monthly_spend    NUMERIC(10, 2) NOT NULL,
    support_tickets  INTEGER NOT NULL,
    last_login_days  INTEGER NOT NULL,
    nps              INTEGER NOT NULL,
    churned          BOOLEAN NOT NULL
);

COPY orders FROM '/seed/orders.csv' WITH (FORMAT csv, HEADER true);
COPY customers FROM '/seed/customers.csv' WITH (FORMAT csv, HEADER true);

-- A read-only role for the app, so connecting as it demonstrates least privilege.
CREATE ROLE analyst LOGIN PASSWORD 'analyst';
GRANT USAGE ON SCHEMA public TO analyst;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO analyst;
