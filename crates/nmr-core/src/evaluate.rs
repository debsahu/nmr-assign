//! Turn a solver result into per-residue chemical shifts and evaluate against ground truth.
//!
//! Provenance: Original implementation of `docs/specs/03-assignment.md` section 5 and
//! `docs/specs/05-benchmark.md`.

use std::collections::BTreeMap;

use crate::nmrstar::Reference;
use crate::priors::Priors;
use crate::solver::{candidate_llr, AssignResult};
use crate::spinsystems::SpinSystem;

/// Convert a spin-system assignment into per-residue shifts and confidences.
/// `i-1` carbons fill the preceding residue when it has no spin system of its own (e.g. before Pro).
pub fn to_reference(
    result: &AssignResult,
    systems: &[SpinSystem],
    sequence: &[String],
    seq_ids: &[i32],
    min_conf: f64,
    priors: Option<&Priors>,
) -> (Reference, BTreeMap<usize, f64>) {
    let default_priors;
    let pri = match priors {
        Some(p) => p,
        None => {
            default_priors = Priors::provisional();
            &default_priors
        }
    };
    let mut out = Reference::new(sequence.to_vec(), seq_ids.to_vec());
    let mut conf: BTreeMap<usize, f64> = BTreeMap::new();
    let mut at: BTreeMap<usize, usize> = BTreeMap::new();
    for (s, &k) in result.position.iter().enumerate() {
        if k >= 0 {
            at.insert(k as usize, s);
        }
    }

    let pick =
        |cands: &[(f64, usize)], res: &str, atom: &str, partner: Option<&[(f64, usize)]>| -> f64 {
            if let Some(part) = partner {
                if !part.is_empty() {
                    return cands
                        .iter()
                        .min_by(|a, b| {
                            let da = part
                                .iter()
                                .map(|p| (a.0 - p.0).abs())
                                .fold(f64::INFINITY, f64::min);
                            let db = part
                                .iter()
                                .map(|p| (b.0 - p.0).abs())
                                .fold(f64::INFINITY, f64::min);
                            da.total_cmp(&db)
                        })
                        .map(|c| c.0)
                        .unwrap_or(cands[0].0);
                }
            }
            let (_, idx) = candidate_llr(pri, res, atom, cands, 0.7);
            cands[idx].0
        };

    for (s, &k_raw) in result.position.iter().enumerate() {
        if k_raw < 0 || result.confidence[s] < min_conf {
            continue;
        }
        let k = k_raw as usize;
        let ss = &systems[s];
        conf.insert(k, result.confidence[s]);
        out.shifts.insert((k, "H".to_string()), ss.h);
        out.shifts.insert((k, "N".to_string()), ss.n);
        let nxt = at.get(&(k + 1)).map(|&ns| &systems[ns]);
        for ((atom, off), cands) in &ss.cands {
            if *off == 0 && !cands.is_empty() {
                let partner = nxt
                    .and_then(|n_ss| n_ss.cands.get(&(atom.clone(), -1)))
                    .map(|v| v.as_slice());
                let val = pick(cands, &sequence[k], atom, partner);
                out.shifts.insert((k, atom.clone()), val);
            }
        }
    }

    for (s, &k_raw) in result.position.iter().enumerate() {
        if k_raw < 1 || result.confidence[s] < min_conf {
            continue;
        }
        let k = k_raw as usize;
        for ((atom, off), cands) in &systems[s].cands {
            if *off == -1 && !cands.is_empty() && !out.shifts.contains_key(&(k - 1, atom.clone())) {
                let val = pick(cands, &sequence[k - 1], atom, None);
                out.shifts.insert((k - 1, atom.clone()), val);
                conf.entry(k - 1).or_insert(result.confidence[s]);
            }
        }
    }

    (out, conf)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Metrics {
    pub assignable: usize,
    pub assigned: usize,
    pub correct: usize,
    pub hc_assigned: usize,
    pub hc_correct: usize,
}

impl Metrics {
    pub fn accuracy(&self) -> f64 {
        if self.assigned > 0 {
            self.correct as f64 / self.assigned as f64
        } else {
            0.0
        }
    }

    pub fn recall(&self) -> f64 {
        if self.assignable > 0 {
            self.correct as f64 / self.assignable as f64
        } else {
            0.0
        }
    }

    pub fn hc_precision(&self) -> f64 {
        if self.hc_assigned > 0 {
            self.hc_correct as f64 / self.hc_assigned as f64
        } else {
            0.0
        }
    }

    pub fn hc_recall(&self) -> f64 {
        if self.assignable > 0 {
            self.hc_correct as f64 / self.assignable as f64
        } else {
            0.0
        }
    }

    pub fn report(&self) -> String {
        format!(
            "assignable spin systems : {}\n\
             assigned                : {}\n\
             accuracy (correct/assigned)   : {:.1}%\n\
             recall   (correct/assignable) : {:.1}%\n\
             high-confidence (>=0.8) precision : {:.1}% ({}/{})\n\
             high-confidence recall            : {:.1}%",
            self.assignable,
            self.assigned,
            self.accuracy() * 100.0,
            self.recall() * 100.0,
            self.hc_precision() * 100.0,
            self.hc_correct,
            self.hc_assigned,
            self.hc_recall() * 100.0
        )
    }
}

pub fn score(result: &AssignResult, truth: &[i32], hc: f64) -> Metrics {
    let assignable = truth.iter().filter(|&&t| t >= 0).count();
    let mut assigned = 0usize;
    let mut correct = 0usize;
    let mut hc_assigned = 0usize;
    let mut hc_correct = 0usize;

    for (s, &pos) in result.position.iter().enumerate() {
        if pos >= 0 {
            assigned += 1;
            let is_ok = pos == truth[s];
            if is_ok {
                correct += 1;
            }
            if result.confidence[s] >= hc {
                hc_assigned += 1;
                if is_ok {
                    hc_correct += 1;
                }
            }
        }
    }

    Metrics {
        assignable,
        assigned,
        correct,
        hc_assigned,
        hc_correct,
    }
}

pub fn compare_shifts(pred: &Reference, ref_data: &Reference) -> BTreeMap<String, (usize, usize)> {
    let tols = [
        ("H", 0.05),
        ("N", 0.50),
        ("CA", 0.30),
        ("CB", 0.40),
        ("C", 0.30),
    ];
    let mut out: BTreeMap<String, (usize, usize)> =
        tols.iter().map(|&(a, _)| (a.to_string(), (0, 0))).collect();

    for ((k, atom), &v) in &pred.shifts {
        let Some(&r) = ref_data.shifts.get(&(*k, atom.clone())) else {
            continue;
        };
        let Some(&(_, tol)) = tols.iter().find(|&&(a, _)| a == atom.as_str()) else {
            continue;
        };
        if let Some(entry) = out.get_mut(atom) {
            entry.1 += 1;
            if (r - v).abs() <= tol {
                entry.0 += 1;
            }
        }
    }
    out
}
