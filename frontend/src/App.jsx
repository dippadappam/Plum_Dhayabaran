import { useEffect, useRef, useState } from "react";
import { SCENARIOS } from "./scenarios";

const API = "/api";

const fmtINR = (n) =>
  n === null || n === undefined ? "—" : "₹" + Number(n).toLocaleString("en-IN");

// ---------------------------------------------------------------------------
// Submit panel: two modes — a real claim (everything read from documents) and
// a sample case (one of the 12 known scenarios).
// ---------------------------------------------------------------------------

function SubmitPanel({ policy, onResult, onSubmitted, parentRef }) {
  const [mode, setMode] = useState("real");

  return (
    <div className="panel">
      <h2>Submit a claim</h2>

      <div className="mode-tabs" role="tablist">
        <button
          role="tab"
          aria-selected={mode === "real"}
          className={`mode-tab ${mode === "real" ? "active" : ""}`}
          onClick={() => setMode("real")}
        >
          Submit a real claim
        </button>
        <button
          role="tab"
          aria-selected={mode === "sample"}
          className={`mode-tab ${mode === "sample" ? "active" : ""}`}
          onClick={() => setMode("sample")}
        >
          Load a sample case
        </button>
      </div>

      {mode === "real" ? (
        <RealClaimMode policy={policy} onResult={onResult}
                       onSubmitted={onSubmitted} parentRef={parentRef} />
      ) : (
        <SampleCaseMode policy={policy} onResult={onResult} onSubmitted={onSubmitted} />
      )}
    </div>
  );
}

// --- Real claim: member + category + bulk document upload only. The amount,
// hospital, and treatment date are read from the uploaded documents. ---

function RealClaimMode({ policy, onResult, onSubmitted, parentRef }) {
  const [memberId, setMemberId] = useState("");
  const [docs, setDocs] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  // The claim type is derived from the documents. Only when the system
  // reports CATEGORY_NEEDED (genuinely ambiguous documents) does a picker
  // appear, and the chosen category is sent on resubmission.
  const [needCategory, setNeedCategory] = useState(false);
  const [category, setCategory] = useState("");
  const fileRef = useRef(null);

  const addFiles = (fileList) => {
    const files = Array.from(fileList || []);
    files.forEach((file) => {
      const reader = new FileReader();
      reader.onload = () => {
        const base64 = String(reader.result).split(",")[1];
        setDocs((d) => [
          ...d,
          {
            file_id: "UP" + Date.now() + "-" + d.length,
            file_name: file.name,
            file_data: base64,
            media_type: file.type || "image/jpeg",
          },
        ]);
      };
      reader.readAsDataURL(file);
    });
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    addFiles(e.dataTransfer.files);
  };

  const submit = async () => {
    setBusy(true);
    setError("");
    const payload = {
      member_id: memberId,
      policy_id: policy?.policy_id || "PLUM_GHI_2024",
      documents: docs.map((d) => ({
        file_id: d.file_id,
        file_name: d.file_name,
        file_data: d.file_data,
        media_type: d.media_type,
      })),
    };
    // Only sent when the system asked for it (ambiguous documents).
    if (needCategory && category) payload.claim_category = category;
    // Link this submission to the claim it is resubmitting, if any.
    if (parentRef) payload.parent_claim_reference = parentRef;
    try {
      const res = await fetch(`${API}/claims`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        const msgs = (detail.detail || []).map((d) => d.msg || "");
        // Unsupported photo format (e.g. iPhone HEIC): show the clear,
        // specific message rather than a generic "submission incomplete".
        const formatMsg = msgs.find((m) => /photo format is not supported/i.test(m));
        throw new Error(
          formatMsg
            ? "This photo format is not supported. Please upload a JPG, PNG, or PDF."
            : res.status === 422
              ? "The submission is incomplete: " +
                (msgs.join("; ") || "check the fields.")
              : `Request failed (${res.status}).`
        );
      }
      const json = await res.json();
      if (json.document_issues?.some((i) => i.issue_code === "CATEGORY_NEEDED")) {
        setNeedCategory(true);
      }
      onResult(json);
      onSubmitted();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const canSubmit = memberId && docs.length > 0 && (!needCategory || category);

  return (
    <>
      <label htmlFor="rc-member">Member</label>
      <select id="rc-member" value={memberId} onChange={(e) => setMemberId(e.target.value)}>
        <option value="">Select member…</option>
        {(policy?.members || []).map((m) => (
          <option key={m.member_id} value={m.member_id}>
            {m.member_id} — {m.name} ({m.relationship})
          </option>
        ))}
      </select>

      {needCategory && (
        <>
          <label htmlFor="rc-category">Claim type</label>
          <select id="rc-category" value={category}
                  onChange={(e) => setCategory(e.target.value)}>
            <option value="">Select the claim type…</option>
            {(policy?.categories || ["consultation"]).map((c) => (
              <option key={c} value={c.toUpperCase()}>{c.replace("_", " ")}</option>
            ))}
          </select>
          <p className="hint">
            Your documents did not clearly indicate a claim type, so the
            system is asking instead of guessing. Pick the type and process
            the claim again — your files are still attached.
          </p>
        </>
      )}

      <label>Documents</label>
      <div
        className={`dropzone ${dragging ? "dragging" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => fileRef.current?.click()}
        role="button"
        tabIndex={0}
      >
        Drop bills, prescriptions, and reports here, or click to select.
        <input
          ref={fileRef}
          type="file"
          accept="image/*,.pdf"
          multiple
          hidden
          onChange={(e) => { addFiles(e.target.files); e.target.value = ""; }}
        />
      </div>
      <p className="hint">
        The system reads the claim type, amount, hospital, treatment date,
        diagnosis, and document types directly from these documents — you do
        not type them. If the documents are ambiguous about the claim type,
        you will be asked to pick one.
      </p>

      {docs.length > 0 && (
        <div className="doc-list">
          {docs.map((d, i) => (
            <div className="doc-chip" key={d.file_id}>
              <span>{d.file_name || d.file_id}</span>
              <button
                onClick={() => setDocs(docs.filter((_, j) => j !== i))}
                aria-label={`Remove ${d.file_name || d.file_id}`}
              >
                Remove
              </button>
            </div>
          ))}
        </div>
      )}

      <button className="btn btn-primary" disabled={!canSubmit || busy} onClick={submit}>
        {busy ? "Processing…" : "Process claim"}
      </button>
      {error && <div className="error-banner" role="alert">{error}</div>}
    </>
  );
}

// --- Sample case: load and run one of the 12 known scenarios verbatim. ---

function SampleCaseMode({ onResult, onSubmitted }) {
  const [scenario, setScenario] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const run = async () => {
    const s = SCENARIOS.find((x) => x.id === scenario);
    if (!s) return;
    setBusy(true);
    setError("");
    try {
      const res = await fetch(`${API}/claims`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...s.payload }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(
          res.status === 422
            ? "The submission is incomplete: " +
              (detail.detail?.map((d) => d.msg).join("; ") || "check the fields.")
            : `Request failed (${res.status}).`
        );
      }
      onResult(await res.json());
      onSubmitted();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <label htmlFor="scenario">Sample case</label>
      <select id="scenario" value={scenario} onChange={(e) => setScenario(e.target.value)}>
        <option value="">Choose one of the 12 known cases…</option>
        {SCENARIOS.map((s) => (
          <option key={s.id} value={s.id}>{s.label}</option>
        ))}
      </select>
      <p className="hint">
        Sample cases carry structured document content and run end-to-end with
        no extraction, reproducing an official test case exactly.
      </p>
      <button className="btn btn-primary" disabled={!scenario || busy} onClick={run}>
        {busy ? "Processing…" : "Run sample case"}
      </button>
      {error && <div className="error-banner" role="alert">{error}</div>}
    </>
  );
}

// ---------------------------------------------------------------------------
// Decision card + audit rail
// ---------------------------------------------------------------------------

function DecisionCard({ result, onClear }) {
  if (!result) {
    return (
      <div className="panel">
        <h2>Decision</h2>
        <p className="empty">
          Submit a claim to see the decision and its complete audit trail.
        </p>
      </div>
    );
  }

  const label = result.decision || "NEEDS_RESUBMISSION";
  const isResubmission = result.status === "NEEDS_RESUBMISSION";
  const b = result.amount_breakdown;

  return (
    <div className={`panel decision-card ${label}`}>
      <div className="decision-head">
        <div>
          <div className={`decision-label ${label}`}>
            {label.replace(/_/g, " ")}
          </div>
          <div className="claim-ref">{result.claim_reference}</div>
        </div>
        {result.approved_amount !== null && result.approved_amount !== undefined && (
          <div>
            <div className="amount">{fmtINR(result.approved_amount)}</div>
            <div className="hint" style={{ textAlign: "right" }}>approved</div>
          </div>
        )}
        {!isResubmission && (
          <div className="confidence">
            <div className="value">{(result.confidence_score ?? 0).toFixed(2)}</div>
            <div className="meter" aria-hidden="true">
              <div style={{ width: `${(result.confidence_score ?? 0) * 100}%` }} />
            </div>
            <div className="hint">confidence</div>
          </div>
        )}
        <button className="btn btn-ghost" onClick={onClear}>New claim</button>
      </div>

      {(result.derived_category || result.review_priority) && (
        <div className="meta-strip">
          {result.derived_category && (
            <span className="meta-chip">
              Category from documents: {result.derived_category.replace(/_/g, " ")}
            </span>
          )}
          {result.review_priority && (
            <span className={`meta-chip priority-${result.review_priority}`}>
              Review priority: {result.review_priority.toUpperCase()}
            </span>
          )}
        </div>
      )}

      {result.reasons?.length > 0 && (
        <ul className="reasons">
          {result.reasons.map((r, i) => <li key={i}>{r}</li>)}
        </ul>
      )}

      {result.document_issues?.length > 0 && (
        <>
          <div className="section-title">What to fix</div>
          {result.document_issues.map((d, i) => (
            <div className="issue" key={i}>
              <div>{d.message}</div>
              <div className="action">→ {d.action_required}</div>
            </div>
          ))}
        </>
      )}

      {result.line_items?.length > 0 && (
        <>
          <div className="section-title">Line items</div>
          <table className="items">
            <thead>
              <tr><th>Item</th><th>Status</th><th>Claimed</th><th>Approved</th><th>Reason</th></tr>
            </thead>
            <tbody>
              {result.line_items.map((li, i) => (
                <tr key={i}>
                  <td>{li.description}</td>
                  <td><span className={`pill ${li.status}`}>{li.status}</span></td>
                  <td className="num">{fmtINR(li.claimed_amount)}</td>
                  <td className="num">{fmtINR(li.approved_amount)}</td>
                  <td>{li.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {b && (
        <>
          <div className="section-title">Amount calculation</div>
          <div className="math-row"><span>Claimed</span><span className="v">{fmtINR(b.claimed_amount)}</span></div>
          <div className="math-row"><span>Eligible after exclusions</span><span className="v">{fmtINR(b.eligible_amount)}</span></div>
          {b.network_discount_percent > 0 && (
            <div className="math-row">
              <span>Network discount ({b.network_discount_percent}%) — applied first</span>
              <span className="v">−{fmtINR(b.network_discount_amount)}</span>
            </div>
          )}
          {b.copay_percent > 0 && (
            <div className="math-row">
              <span>Co-pay ({b.copay_percent}%)</span>
              <span className="v">−{fmtINR(b.copay_amount)}</span>
            </div>
          )}
          <div className="math-row total"><span>Approved</span><span className="v">{fmtINR(b.approved_amount)}</span></div>
        </>
      )}

      {result.fraud_signals?.length > 0 && (
        <>
          <div className="section-title">Fraud signals</div>
          {result.fraud_signals.map((s, i) => <div className="signal" key={i}>{s}</div>)}
        </>
      )}

      {result.component_failures?.length > 0 && (
        <>
          <div className="section-title">Component failures</div>
          {result.component_failures.map((c, i) => (
            <div className="failure" key={i}>
              <strong>{c.component}</strong>: {c.error} {c.impact}
            </div>
          ))}
        </>
      )}

      <div className="section-title">Audit trail — every check, in order</div>
      <div className="rail">
        {result.trace?.map((s, i) => (
          <div className="rail-step" key={i}>
            <span className={`node ${s.status}`} aria-hidden="true" />
            <div className="meta">
              {s.stage} / {s.check} — <span className={`status ${s.status}`}>{s.status}</span>
            </div>
            <div className="detail">{s.detail}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Review queue
// ---------------------------------------------------------------------------

function Queue({ refreshKey, onOpen }) {
  const [rows, setRows] = useState([]);
  useEffect(() => {
    fetch(`${API}/claims`).then((r) => r.json()).then(setRows).catch(() => {});
  }, [refreshKey]);

  return (
    <div className="queue">
      <div className="panel">
        <h2>Review queue — recent claims</h2>
        {rows.length === 0 ? (
          <p className="empty">No claims processed yet.</p>
        ) : (
          <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Reference</th><th>Member</th><th>Category</th>
                <th>Treatment date</th><th>Claimed</th><th>Decision</th>
                <th>Approved</th><th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.claim_reference} className="clickable"
                    onClick={() => onOpen(r.claim_reference)}>
                  <td className="mono">{r.claim_reference}</td>
                  <td>{r.member_id}</td>
                  <td>{r.claim_category}</td>
                  <td>{r.treatment_date}</td>
                  <td className="mono">{fmtINR(r.claimed_amount)}</td>
                  <td>{r.decision || r.status}</td>
                  <td className="mono">{fmtINR(r.approved_amount)}</td>
                  <td className="mono">{r.confidence?.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Held-for-review queue (actionable): the human side of the auto→hold→human loop
// ---------------------------------------------------------------------------

function ReviewQueue({ refreshKey, onResolved }) {
  const [held, setHeld] = useState([]);
  const [reasons, setReasons] = useState({});
  const [amounts, setAmounts] = useState({});
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  const load = () =>
    fetch(`${API}/review-queue`)
      .then((r) => r.json())
      .then((d) => setHeld(Array.isArray(d) ? d : []))
      .catch(() => {});
  useEffect(() => { load(); }, [refreshKey]);

  const resolve = async (ref, action) => {
    const reason = (reasons[ref] || "").trim();
    if (!reason) { setError("A reason is required to resolve a claim."); return; }
    const body = { action, reviewer_id: "reviewer-demo", reason };
    const amt = amounts[ref];
    if (action === "approve" && amt !== undefined && amt !== "") {
      body.approved_amount = Number(amt);
    }
    setBusy(ref); setError("");
    try {
      const res = await fetch(`${API}/claims/${ref}/resolve`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const d = await res.json().catch(() => ({}));
        throw new Error(d.detail || `Resolve failed (${res.status}).`);
      }
      await load();
      onResolved && onResolved();
    } catch (e) { setError(e.message); } finally { setBusy(""); }
  };

  if (held.length === 0) return null;

  return (
    <div className="queue">
      <div className="panel">
        <h2>Held for review — {held.length} awaiting a human</h2>
        {error && <div className="error-banner" role="alert">{error}</div>}
        <div className="held-list">
          {held.map((c) => (
            <details className="held-claim" key={c.claim_reference}>
              <summary className="held-head">
                <span className="mono">{c.claim_reference}</span>
                <span>{c.member_id} · {c.claim_category}</span>
                {c.review_priority && (
                  <span className={`meta-chip priority-${c.review_priority}`}>
                    {c.review_priority.toUpperCase()}
                  </span>
                )}
                <span className="mono amount-tag">{fmtINR(c.approved_amount)}</span>
              </summary>
              <div className="held-body">
                <ul className="reasons">
                  {(c.reasons || []).map((r, i) => <li key={`r${i}`}>{r}</li>)}
                  {(c.fraud_signals || []).map((s, i) => <li key={`f${i}`}>{s}</li>)}
                </ul>
                {(c.extracted_documents || []).length > 0 && (
                  <p className="hint">
                    Original documents:{" "}
                    {c.extracted_documents.map((d) => (
                      <a key={d.file_id}
                         href={`${API}/claims/${c.claim_reference}/documents/${d.file_id}`}
                         target="_blank" rel="noreferrer" style={{ marginRight: 10 }}>
                        {d.file_id} ({d.type})
                      </a>
                    ))}
                  </p>
                )}
                {(c.approved_amount === null || c.approved_amount === undefined) && (
                  <input
                    type="number" min="1"
                    placeholder="Approved amount — read the bill and enter it (required to approve)"
                    value={amounts[c.claim_reference] || ""}
                    onChange={(e) =>
                      setAmounts({ ...amounts, [c.claim_reference]: e.target.value })}
                  />
                )}
                <input
                  placeholder="Reason for your decision (required)"
                  value={reasons[c.claim_reference] || ""}
                  onChange={(e) =>
                    setReasons({ ...reasons, [c.claim_reference]: e.target.value })}
                />
                <div className="scenario-row" style={{ marginTop: 8 }}>
                  <button className="btn btn-ghost" disabled={busy === c.claim_reference}
                          onClick={() => resolve(c.claim_reference, "approve")}>Approve</button>
                  <button className="btn btn-ghost" disabled={busy === c.claim_reference}
                          onClick={() => resolve(c.claim_reference, "reject")}>Reject</button>
                  <button className="btn btn-ghost" disabled={busy === c.claim_reference}
                          onClick={() => resolve(c.claim_reference, "close")}>Close</button>
                </div>
              </div>
            </details>
          ))}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------

export default function App() {
  const [policy, setPolicy] = useState(null);
  const [result, setResult] = useState(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [formKey, setFormKey] = useState(0);

  useEffect(() => {
    fetch(`${API}/policy`).then((r) => r.json()).then(setPolicy).catch(() => {});
  }, []);

  const openClaim = (ref) =>
    fetch(`${API}/claims/${ref}`).then((r) => r.json()).then(setResult);

  const clearAll = () => {
    setResult(null);
    setFormKey((k) => k + 1); // remount the submit panel to reset the form
  };

  return (
    <div className="app">
      <header className="masthead">
        <h1>Claims Processing</h1>
        <span className="policy-tag">
          {policy ? `${policy.policy_id} · ${policy.insurer}` : "loading policy…"}
        </span>
      </header>

      <div className="layout">
        <SubmitPanel
          key={formKey}
          policy={policy}
          onResult={setResult}
          onSubmitted={() => setRefreshKey((k) => k + 1)}
          parentRef={result?.status === "NEEDS_RESUBMISSION"
            ? result.claim_reference : null}
        />
        <DecisionCard result={result} onClear={clearAll} />
      </div>

      <div className="queues">
        <ReviewQueue refreshKey={refreshKey}
                     onResolved={() => setRefreshKey((k) => k + 1)} />
        <Queue refreshKey={refreshKey} onOpen={openClaim} />
      </div>
    </div>
  );
}
