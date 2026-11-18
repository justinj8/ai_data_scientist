import type { TableBlock } from "../api/types";

function fmt(v: unknown): string {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") {
    if (Number.isInteger(v)) return v.toLocaleString();
    return Math.abs(v) >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 2 }) : String(+v.toPrecision(6));
  }
  return String(v);
}

export function DataTable({ block, maxHeight = 320 }: { block: TableBlock; maxHeight?: number }) {
  const numeric = block.columns.map((_, i) => block.rows.some((r) => typeof r[i] === "number"));
  return (
    <div>
      <div className="overflow-auto rounded-md border border-rule bg-surface" style={{ maxHeight }}>
        <table className="tabular w-full border-collapse text-[13px]">
          <thead className="sticky top-0 bg-sunken">
            <tr>
              {block.columns.map((c, i) => (
                <th
                  key={c + i}
                  scope="col"
                  className={`whitespace-nowrap border-b border-rule px-3 py-1.5 font-medium text-ink-2 ${numeric[i] ? "text-right" : "text-left"}`}
                >
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {block.rows.map((row, r) => (
              <tr key={r} className="border-b border-rule/60 last:border-0">
                {row.map((v, i) => (
                  <td key={i} className={`whitespace-nowrap px-3 py-1 ${numeric[i] ? "text-right" : "text-left"}`}>
                    {fmt(v)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {block.total_rows > block.rows.length && (
        <p className="mt-1 text-xs text-ink-3">
          Showing {block.rows.length.toLocaleString()} of {block.total_rows.toLocaleString()} rows
        </p>
      )}
    </div>
  );
}
