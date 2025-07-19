BEGIN;

CREATE TABLE alembic_version (
    version_num VARCHAR(32) NOT NULL, 
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);

-- Running upgrade  -> 8900bcac251d

CREATE TABLE images (
    id SERIAL NOT NULL, 
    receipt_id INTEGER NOT NULL, 
    image_path VARCHAR(255) NOT NULL, 
    uploaded_at TIMESTAMP WITHOUT TIME ZONE, 
    PRIMARY KEY (id), 
    FOREIGN KEY(receipt_id) REFERENCES receipts (id)
);

CREATE INDEX ix_images_id ON images (id);

ALTER TABLE receipts ADD COLUMN unique_key VARCHAR(255) NOT NULL;

ALTER TABLE receipts ADD COLUMN purchase_date TIMESTAMP WITHOUT TIME ZONE;

ALTER TABLE receipts ADD COLUMN total_cost FLOAT NOT NULL;

ALTER TABLE receipt_items ADD COLUMN name VARCHAR(255) NOT NULL;

ALTER TABLE receipt_items ADD COLUMN price_per_item FLOAT NOT NULL;

CREATE INDEX ix_receipts_id ON receipts (id);

CREATE INDEX ix_receipt_items_id ON receipt_items (id);

CREATE INDEX ix_stores_id ON stores (id);

CREATE INDEX ix_users_id ON users (id);

INSERT INTO alembic_version (version_num) VALUES ('8900bcac251d') RETURNING alembic_version.version_num;

COMMIT;

