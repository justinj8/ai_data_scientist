import type { ConnectionConfig, Engine } from "../../api/types";

export type FieldSpec = {
  key: keyof ConnectionConfig;
  label: string;
  type?: "text" | "password" | "number" | "checkbox" | "textarea" | "file";
  placeholder?: string;
  hint?: string;
};

const NETWORK: FieldSpec[] = [
  { key: "host", label: "Host", placeholder: "db.example.com" },
  { key: "port", label: "Port", type: "number" },
  { key: "database", label: "Database" },
  { key: "username", label: "Username" },
  { key: "password", label: "Password", type: "password" },
  { key: "ssl", label: "Require SSL", type: "checkbox" },
];

export const ENGINES: { engine: Engine; label: string; defaultPort?: number; fields: FieldSpec[]; urlExample: string }[] = [
  { engine: "postgresql", label: "PostgreSQL", defaultPort: 5432, fields: NETWORK, urlExample: "postgresql+psycopg://user:pass@host:5432/db" },
  { engine: "mysql", label: "MySQL / MariaDB", defaultPort: 3306, fields: NETWORK, urlExample: "mysql+pymysql://user:pass@host:3306/db" },
  {
    engine: "sqlite",
    label: "SQLite",
    fields: [{ key: "database", label: "Database file path", placeholder: "/data/shop.db", hint: "A path on the machine running the backend." }],
    urlExample: "sqlite:////absolute/path/to/file.db",
  },
  { engine: "mssql", label: "SQL Server", defaultPort: 1433, fields: NETWORK.filter((f) => f.key !== "ssl"), urlExample: "mssql+pymssql://user:pass@host:1433/db" },
  {
    engine: "snowflake",
    label: "Snowflake",
    fields: [
      { key: "account", label: "Account identifier", placeholder: "myorg-myaccount" },
      { key: "username", label: "Username" },
      { key: "password", label: "Password", type: "password" },
      { key: "warehouse", label: "Warehouse" },
      { key: "database", label: "Database" },
      { key: "db_schema", label: "Schema", placeholder: "PUBLIC" },
      { key: "role", label: "Role" },
    ],
    urlExample: "snowflake://user:pass@account/db/schema?warehouse=WH",
  },
  {
    engine: "bigquery",
    label: "BigQuery",
    fields: [
      { key: "project", label: "Project ID" },
      { key: "dataset", label: "Dataset" },
      { key: "credentials_json", label: "Service account key (JSON)", type: "file", hint: "Read in your browser and sent only to your local backend." },
    ],
    urlExample: "bigquery://project/dataset",
  },
];

export function buildConfig(engine: Engine, values: Record<string, string>, useUrl = false): ConnectionConfig {
  if (useUrl) return { engine, url: values.url?.trim() };
  const spec = ENGINES.find((e) => e.engine === engine)!;
  const config: ConnectionConfig = { engine };
  for (const f of spec.fields) {
    const raw = values[f.key]?.trim();
    if (!raw) continue;
    if (f.type === "number") (config as Record<string, unknown>)[f.key] = Number(raw);
    else if (f.type === "checkbox") (config as Record<string, unknown>)[f.key] = raw === "true";
    else (config as Record<string, unknown>)[f.key] = raw;
  }
  return config;
}

export function suggestName(engine: Engine, values: Record<string, string>, taken: string[]): string {
  const base =
    (values.database || values.dataset || engine)
      .split("/")
      .pop()!
      .replace(/\.[a-z]+$/i, "")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "")
      .replace(/^(\d)/, "db_$1") || engine;
  let name = base.slice(0, 36);
  for (let i = 2; taken.includes(name); i++) name = `${base.slice(0, 36)}_${i}`;
  return name;
}
