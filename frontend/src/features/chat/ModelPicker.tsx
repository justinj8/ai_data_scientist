import { useQuery } from "@tanstack/react-query";
import { api } from "../../api/client";

/** Choose the model: the four latest-model choices, plus every tool-capable model from connected providers. */
export function ModelPicker({
  value,
  onChange,
  onOpenSettings,
}: {
  value: string;
  onChange: (v: string) => void;
  onOpenSettings: () => void;
}) {
  const models = useQuery({ queryKey: ["models"], queryFn: api.models, staleTime: 60_000 });
  const all = useQuery({ queryKey: ["models-all"], queryFn: api.allModels, staleTime: 60_000 });
  const choices = models.data?.choices ?? [];
  const current = choices.find((c) => c.key === value);
  const noneAvailable = choices.length > 0 && !choices.some((c) => c.available);

  return (
    <div className="flex items-center gap-2">
      <label htmlFor="model-picker" className="sr-only">Model</label>
      <select
        id="model-picker"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="max-w-[16rem] rounded-md border border-rule bg-surface px-2 py-1.5 text-sm"
      >
        {choices.map((c) => (
          <option key={c.key} value={c.key} disabled={!c.available}>
            {c.label} ({c.model.id}){c.available ? "" : " — needs credentials"}
            {c.billed_to && c.billed_to !== "api_key" ? " — plan" : ""}
          </option>
        ))}
        {(all.data?.models.length ?? 0) > 0 && (
          <optgroup label="Other models">
            {all.data!.models.map((m) => (
              <option key={m.id} value={m.id}>{m.display_name === m.id ? m.id : `${m.display_name} (${m.id})`}</option>
            ))}
          </optgroup>
        )}
        {!current && !choices.length && <option value={value}>{value}</option>}
      </select>
      {noneAvailable && (
        <button type="button" onClick={onOpenSettings} className="text-sm font-medium text-accent">
          Add a model
        </button>
      )}
    </div>
  );
}
