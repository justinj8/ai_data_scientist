import { X } from "lucide-react";
import { useEffect, useRef, type ReactNode } from "react";

/** Accessible modal built on <dialog> (focus trap and Escape handling come from the browser). */
export function Dialog({
  open,
  onClose,
  title,
  children,
  width = "36rem",
}: {
  open: boolean;
  onClose: () => void;
  title: string;
  children: ReactNode;
  width?: string;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);
  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onClick={(e) => e.target === ref.current && onClose()}
      className="m-auto max-h-[90vh] w-[calc(100%-2rem)] rounded-xl border border-rule bg-surface p-0 text-ink shadow-2xl backdrop:bg-[#0b1220]/45"
      style={{ maxWidth: width }}
    >
      {open && (
        <div className="flex max-h-[90vh] flex-col">
          <div className="flex items-center justify-between border-b border-rule px-5 py-3">
            <h2 className="text-base font-semibold">{title}</h2>
            <button type="button" onClick={onClose} className="rounded p-1 text-ink-3 hover:bg-sunken" aria-label="Close">
              <X size={18} />
            </button>
          </div>
          <div className="overflow-y-auto px-5 py-4">{children}</div>
        </div>
      )}
    </dialog>
  );
}

export const inputClass =
  "w-full rounded-md border border-rule bg-surface px-2.5 py-1.5 text-sm placeholder:text-ink-3 focus:border-accent focus:outline-none";
export const primaryButton =
  "inline-flex shrink-0 items-center justify-center gap-1.5 whitespace-nowrap rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:brightness-110 disabled:opacity-50 dark:text-[#0b1a1d]";
export const secondaryButton =
  "inline-flex shrink-0 items-center justify-center gap-1.5 whitespace-nowrap rounded-md border border-rule bg-surface px-3 py-1.5 text-sm hover:bg-sunken disabled:opacity-50";
