import { Check, Copy } from "lucide-react";
import { Highlight, themes } from "prism-react-renderer";
import { useState } from "react";

export function CodeBlock({ code, language }: { code: string; language: "sql" | "python" | string }) {
  const [copied, setCopied] = useState(false);
  const dark = document.documentElement.dataset.theme === "dark";
  return (
    <div className="group relative">
      <Highlight code={code.trimEnd()} language={language} theme={dark ? themes.nightOwl : themes.github}>
        {({ tokens, getLineProps, getTokenProps }) => (
          <pre className="overflow-x-auto rounded-md border border-rule bg-sunken px-3 py-2.5 font-mono text-[12.5px] leading-relaxed">
            {tokens.map((line, i) => (
              <div key={i} {...getLineProps({ line })} style={undefined}>
                {line.map((token, k) => {
                  const props = getTokenProps({ token });
                  return <span key={k} {...props} style={{ color: props.style?.color }} />;
                })}
              </div>
            ))}
          </pre>
        )}
      </Highlight>
      <button
        type="button"
        onClick={() => {
          void navigator.clipboard.writeText(code);
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        }}
        className="absolute right-2 top-2 rounded p-1 text-ink-3 opacity-0 transition-opacity hover:text-ink focus-visible:opacity-100 group-hover:opacity-100"
        aria-label={copied ? "Copied" : "Copy code"}
      >
        {copied ? <Check size={14} /> : <Copy size={14} />}
      </button>
    </div>
  );
}
