import { describe, expect, it } from "vitest";
import { buildConfig, ENGINES, suggestName } from "./connectionForm";

describe("buildConfig", () => {
  it("builds a Postgres config with a numeric port and drops empty fields", () => {
    expect(
      buildConfig("postgresql", { host: "db.local", port: "5433", database: "shop", username: "me", password: "pw", ssl: "true", account: "" }),
    ).toEqual({ engine: "postgresql", host: "db.local", port: 5433, database: "shop", username: "me", password: "pw", ssl: true });
  });

  it("builds Snowflake and BigQuery configs from their own fields only", () => {
    expect(
      buildConfig("snowflake", { account: "org-acct", username: "u", password: "p", warehouse: "WH", database: "DB", db_schema: "PUBLIC", host: "ignored" }),
    ).toEqual({ engine: "snowflake", account: "org-acct", username: "u", password: "p", warehouse: "WH", database: "DB", db_schema: "PUBLIC" });
    expect(buildConfig("bigquery", { project: "p", dataset: "d", credentials_json: '{"type":"service_account"}' })).toEqual({
      engine: "bigquery", project: "p", dataset: "d", credentials_json: '{"type":"service_account"}',
    });
  });

  it("uses only the URL when the URL option is on", () => {
    expect(buildConfig("mysql", { url: "mysql+pymysql://u:p@h/db", host: "h" }, true)).toEqual({ engine: "mysql", url: "mysql+pymysql://u:p@h/db" });
  });

  it("lists every supported engine with a field spec", () => {
    expect(ENGINES.map((e) => e.engine)).toEqual(["postgresql", "mysql", "sqlite", "mssql", "snowflake", "bigquery"]);
  });

  it("suggests a valid, unused source name", () => {
    expect(suggestName("postgresql", { database: "Retail Prod" }, ["retail_prod"])).toBe("retail_prod_2");
    expect(suggestName("bigquery", {}, [])).toBe("bigquery");
  });
});
