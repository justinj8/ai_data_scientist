import { useQuery } from "@tanstack/react-query";
import { api } from "../../api/client";
import type { Provider } from "../../api/types";
import { Dialog } from "../../components/Dialog";
import { ProviderCard } from "./ProviderCard";

const ORDER: Provider[] = ["anthropic", "openai", "xai"];

export function SettingsDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const creds = useQuery({ queryKey: ["credentials"], queryFn: api.credentials, enabled: open });
  return (
    <Dialog open={open} onClose={onClose} title="Models and credentials" width="40rem">
      <p className="mb-4 max-w-[62ch] text-sm text-ink-2">
        Connect at least one provider. Keys and sign-ins are stored encrypted on this machine and sent only to their own
        provider. The model picker uses the newest Opus, Sonnet, GPT and Grok models your credentials can reach.
      </p>
      {creds.error && <p className="text-sm text-bad">{creds.error.message}</p>}
      <div className="space-y-3">
        {creds.data && ORDER.map((p) => <ProviderCard key={p} provider={p} status={creds.data.providers[p]} />)}
      </div>
    </Dialog>
  );
}
