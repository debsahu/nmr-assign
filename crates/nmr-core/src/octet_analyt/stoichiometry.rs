//! Reaction perception (classifying 2D canvas molecules and captions relative to reaction arrows
//! into Reactants, Above/Below-Arrow Reagents/Catalysts/Solvents, and Products), atom-balance
//! verification, and complete laboratory Stoichiometry Table engine.
//!
//! Provenance & Licence:
//! Ported/adapted from Octet-Draw by Sam Ellis (<https://github.com/SamChemMan/Octet-Draw>, Apache-2.0)
//! and published chemical shift additivity / IUPAC nomenclature rules (`stoich.ts`, `reaction.ts`,
//! `interpret.ts`).

use crate::octet_analyt::mass_props::{
    element_counts_of_mol, hill_formula, molecular_weight, parse_formula_counts, ElementCounts,
};
use crate::octet_analyt::AnalytMol;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// 2D axis-aligned bounding box `[left, bottom, right, top]` (`y` up, in bond-length or canvas units).
pub type CanvasBox = [f64; 4];

/// Straight reaction arrow on a 2D chemical drawing canvas.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReactionArrow {
    pub id: usize,
    pub from: (f64, f64),
    pub to: (f64, f64),
}

/// Text caption on a 2D chemical drawing canvas (`+`, reagent formula/condition, or `% yield`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReactionCaption {
    pub id: usize,
    pub text: String,
    pub bbox: CanvasBox,
    /// Optional explicit attachment to `(arrow_id, is_above)`.
    pub pinned_on_arrow: Option<(usize, bool)>,
}

/// Role of a molecule or text item relative to a reaction arrow (`interpret.ts`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReactionRole {
    Reactant,
    ReagentAbove,
    ReagentBelow,
    Product,
}

impl ReactionRole {
    pub fn is_reagent_or_condition(&self) -> bool {
        matches!(
            self,
            ReactionRole::ReagentAbove | ReactionRole::ReagentBelow
        )
    }
}

/// Classified molecule component in a perceived reaction step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceivedMolecule {
    pub mol_index: usize,
    pub role: ReactionRole,
    pub formula: String,
    pub mw: f64,
    pub bbox: CanvasBox,
    /// Optional attached yield caption (e.g. `88.0` from `"88% yield"`).
    pub attached_yield_percent: Option<f64>,
}

/// Classified reagent / catalyst / solvent / condition text item on or near an arrow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceivedConditionText {
    pub caption_id: usize,
    pub role: ReactionRole,
    pub raw_text: String,
    /// Individual comma/semicolon/newline-separated items (keeping parenthesized ratios intact).
    pub items: Vec<String>,
    /// True if all items represent an inert atmosphere (`N2`, `Ar`, `air`, etc.).
    pub is_inert_atmosphere: bool,
}

/// Single perceived reaction step around one `ReactionArrow`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerceivedReactionStep {
    pub arrow_id: usize,
    pub reactants: Vec<PerceivedMolecule>,
    pub reagents_above: Vec<PerceivedMolecule>,
    pub reagents_below: Vec<PerceivedMolecule>,
    pub condition_texts: Vec<PerceivedConditionText>,
    pub products: Vec<PerceivedMolecule>,
}

/// Atom-balance check result comparing total reactant element counts against total product element counts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtomBalanceReport {
    pub balanced: bool,
    pub reactant_counts: ElementCounts,
    pub product_counts: ElementCounts,
    /// Net element difference `product_count - reactant_count` for any unbalanced element.
    pub delta_counts: BTreeMap<String, isize>,
    /// Human-readable missing/excess summary (e.g., `"-H2O"` for condensation loss of water).
    pub summary: String,
}

fn mol_bounding_box(mol: &AnalytMol) -> CanvasBox {
    if mol.atoms.is_empty() {
        return [0.0, 0.0, 0.0, 0.0];
    }
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for a in &mol.atoms {
        min_x = min_x.min(a.x);
        max_x = max_x.max(a.x);
        min_y = min_y.min(a.y);
        max_y = max_y.max(a.y);
    }
    [min_x, min_y, max_x, max_y]
}

fn box_gap(p: &CanvasBox, q: &CanvasBox) -> f64 {
    (p[0] - q[2])
        .max(q[0] - p[2])
        .max(p[1] - q[3])
        .max(q[1] - p[3])
        .max(0.0)
}

/// Extract numeric percentage if `text` is a reaction yield annotation (`"85%"`, `"(92% yield)"`).
pub fn parse_yield_caption(text: &str) -> Option<f64> {
    let s = text
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .trim();
    if s.is_empty() {
        return None;
    }
    let s_low = s.to_ascii_lowercase();
    let stripped = s_low
        .strip_prefix("yield:")
        .or_else(|| s_low.strip_prefix("yield"))
        .unwrap_or(&s_low)
        .trim();
    let pct_pos = stripped.find('%')?;
    let num_part = stripped[..pct_pos].trim().replace(',', ".");
    let after = stripped[pct_pos + 1..].trim();
    if !after.is_empty()
        && !matches!(
            after,
            "yield" | "ee" | "de" | "dr" | "conv" | "conv." | "conversion"
        )
    {
        return None;
    }
    let val = num_part.parse::<f64>().ok()?;
    if (0.0..=200.0).contains(&val) {
        Some(val)
    } else {
        None
    }
}

/// Split a reaction condition line into items at commas, semicolons, and newlines outside brackets (`conditionItems` in `stoich.ts`).
pub fn split_condition_items(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut depth = 0usize;
    let mut cur = String::new();
    for ch in text.chars() {
        if matches!(ch, '(' | '[' | '{') {
            depth += 1;
        } else if matches!(ch, ')' | ']' | '}') {
            depth = depth.saturating_sub(1);
        }
        if depth == 0 && matches!(ch, ',' | ';' | '\n') {
            let trimmed = cur.trim();
            if !trimmed.is_empty() {
                out.push(trimmed.to_string());
            }
            cur.clear();
        } else {
            cur.push(ch);
        }
    }
    let trimmed = cur.trim();
    if !trimmed.is_empty() {
        out.push(trimmed.to_string());
    }
    out
}

/// True when `text` names an inert atmosphere (`N2`, `N₂`, `Ar`, `argon`, `air`, `under N2 (balloon)`),
/// which is excluded from stoichiometry reagent columns (`inert` in `stoich.ts`).
pub fn is_inert_gas_condition(text: &str) -> bool {
    let mut s = text.trim().to_ascii_lowercase();
    if let Some(rest) = s.strip_prefix("under ") {
        s = rest.trim().to_string();
    }
    for suffix in ["(atm)", "atm", "(balloon)", "balloon"] {
        if let Some(rest) = s.strip_suffix(suffix) {
            s = rest.trim().to_string();
        }
    }
    matches!(
        s.as_str(),
        "n2" | "n₂" | "ar" | "argon" | "nitrogen" | "he" | "helium" | "air"
    )
}

/// Perceive reaction roles (Reactants, Above/Below-Arrow Reagents/Catalysts/Solvents, and Products)
/// for canvas molecules and text captions relative to `arrows` (`reaction.ts` & `interpret.ts`).
pub fn perceive_reaction_canvas(
    molecules: &[AnalytMol],
    captions: &[ReactionCaption],
    arrows: &[ReactionArrow],
) -> Vec<PerceivedReactionStep> {
    if arrows.is_empty() {
        return Vec::new();
    }

    let mol_infos: Vec<(CanvasBox, String, f64)> = molecules
        .iter()
        .map(|m| {
            let bbox = mol_bounding_box(m);
            let (cts, _) = element_counts_of_mol(m);
            let f = hill_formula(&cts);
            let mw = molecular_weight(&cts).unwrap_or(0.0);
            (bbox, f, mw)
        })
        .collect();

    // Attach yield captions (`"86% yield"`) to the nearest molecule within reasonable canvas distance
    let mut yield_for_mol: BTreeMap<usize, f64> = BTreeMap::new();
    let mut yield_caption_ids: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for cap in captions {
        if cap.pinned_on_arrow.is_some() {
            continue;
        }
        if let Some(y_pct) = parse_yield_caption(&cap.text) {
            if let Some((best_m, dist)) = mol_infos
                .iter()
                .enumerate()
                .map(|(mi, (mb, _, _))| (mi, box_gap(&cap.bbox, mb)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                if dist <= 4.0 {
                    yield_for_mol.insert(best_m, y_pct);
                    yield_caption_ids.insert(cap.id);
                }
            }
        }
    }

    // Sort arrows along primary reaction flow (left-to-right or top-to-bottom)
    let mut sorted_arrows = arrows.to_vec();
    sorted_arrows.sort_by(|a, b| {
        let ax = 0.5 * (a.from.0 + a.to.0);
        let bx = 0.5 * (b.from.0 + b.to.0);
        ax.partial_cmp(&bx).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut steps = Vec::with_capacity(sorted_arrows.len());

    for (step_idx, arr) in sorted_arrows.iter().enumerate() {
        let dx = arr.to.0 - arr.from.0;
        let dy = arr.to.1 - arr.from.1;
        let len = dx.hypot(dy).max(1e-6);
        let ux = dx / len;
        let uy = dy / len;
        // Normal vector pointing "above" the arrow (`y` up in standard chemical canvas: `(-uy, ux)` for rightward arrow)
        let nx = -uy;
        let ny = ux;
        let ox = 0.5 * (arr.from.0 + arr.to.0);
        let oy = 0.5 * (arr.from.1 + arr.to.1);
        let half_len = 0.5 * len;

        let along = |x: f64, y: f64| (x - ox) * ux + (y - oy) * uy;
        let across = |x: f64, y: f64| (x - ox) * nx + (y - oy) * ny;

        // Multi-step boundary along axis if multiple sequential arrows exist
        let prev_arrow_end = if step_idx > 0 {
            let pa = &sorted_arrows[step_idx - 1];
            along(pa.to.0, pa.to.1)
        } else {
            f64::NEG_INFINITY
        };
        let next_arrow_start = if step_idx + 1 < sorted_arrows.len() {
            let na = &sorted_arrows[step_idx + 1];
            along(na.from.0, na.from.1)
        } else {
            f64::INFINITY
        };

        let mut reactants = Vec::new();
        let mut reagents_above = Vec::new();
        let mut reagents_below = Vec::new();
        let mut products = Vec::new();

        for (mi, (mb, formula, mw)) in mol_infos.iter().enumerate() {
            let cx = 0.5 * (mb[0] + mb[2]);
            let cy = 0.5 * (mb[1] + mb[3]);
            let s_along = along(cx, cy);
            let s_across = across(cx, cy);

            // Project box corners across the arrow line to check if wholly above or wholly below within arrow span
            let corners = [
                (mb[0], mb[1]),
                (mb[2], mb[1]),
                (mb[2], mb[3]),
                (mb[0], mb[3]),
            ];
            let ys: Vec<f64> = corners.iter().map(|&(px, py)| across(px, py)).collect();
            let min_y = ys.iter().copied().fold(f64::INFINITY, f64::min);
            let max_y = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);

            let over_shaft = s_along >= -half_len - 0.5 && s_along <= half_len + 0.5;
            let role = if over_shaft && min_y > 0.05 {
                ReactionRole::ReagentAbove
            } else if over_shaft && max_y < -0.05 {
                ReactionRole::ReagentBelow
            } else if over_shaft && s_across.abs() > 0.25 {
                if s_across > 0.0 {
                    ReactionRole::ReagentAbove
                } else {
                    ReactionRole::ReagentBelow
                }
            } else if s_along < -half_len * 0.2 && s_along >= prev_arrow_end {
                ReactionRole::Reactant
            } else if s_along > half_len * 0.2 && s_along <= next_arrow_start {
                ReactionRole::Product
            } else {
                continue;
            };

            let pm = PerceivedMolecule {
                mol_index: mi,
                role,
                formula: formula.clone(),
                mw: *mw,
                bbox: *mb,
                attached_yield_percent: yield_for_mol.get(&mi).copied(),
            };
            match role {
                ReactionRole::Reactant => reactants.push(pm),
                ReactionRole::ReagentAbove => reagents_above.push(pm),
                ReactionRole::ReagentBelow => reagents_below.push(pm),
                ReactionRole::Product => products.push(pm),
            }
        }

        let mut condition_texts = Vec::new();
        for cap in captions {
            let trimmed = cap.text.trim();
            if trimmed.is_empty() || trimmed == "+" || yield_caption_ids.contains(&cap.id) {
                continue;
            }
            let role_opt = if let Some((aid, is_above)) = cap.pinned_on_arrow {
                if aid == arr.id {
                    Some(if is_above {
                        ReactionRole::ReagentAbove
                    } else {
                        ReactionRole::ReagentBelow
                    })
                } else {
                    None
                }
            } else {
                let cx = 0.5 * (cap.bbox[0] + cap.bbox[2]);
                let cy = 0.5 * (cap.bbox[1] + cap.bbox[3]);
                let sa = along(cx, cy);
                let sc = across(cx, cy);
                if sa >= -half_len - 1.0 && sa <= half_len + 1.0 && sc.abs() <= 6.0 {
                    Some(if sc >= 0.0 {
                        ReactionRole::ReagentAbove
                    } else {
                        ReactionRole::ReagentBelow
                    })
                } else {
                    None
                }
            };

            if let Some(role) = role_opt {
                let items = split_condition_items(trimmed);
                let is_inert = !items.is_empty() && items.iter().all(|x| is_inert_gas_condition(x));
                condition_texts.push(PerceivedConditionText {
                    caption_id: cap.id,
                    role,
                    raw_text: trimmed.to_string(),
                    items,
                    is_inert_atmosphere: is_inert,
                });
            }
        }

        steps.push(PerceivedReactionStep {
            arrow_id: arr.id,
            reactants,
            reagents_above,
            reagents_below,
            condition_texts,
            products,
        });
    }

    steps
}

/// Check elemental atom balance between `reactants` (with optional stoichiometric coefficients)
/// and `products` (with optional stoichiometric coefficients).
pub fn check_reaction_balance(
    reactants: &[(&AnalytMol, usize)],
    products: &[(&AnalytMol, usize)],
) -> AtomBalanceReport {
    let mut r_counts = ElementCounts::new();
    let mut p_counts = ElementCounts::new();

    for &(m, coeff) in reactants {
        let c = coeff.max(1);
        let (cts, _) = element_counts_of_mol(m);
        for (el, n) in cts {
            *r_counts.entry(el).or_insert(0) += n * c;
        }
    }
    for &(m, coeff) in products {
        let c = coeff.max(1);
        let (cts, _) = element_counts_of_mol(m);
        for (el, n) in cts {
            *p_counts.entry(el).or_insert(0) += n * c;
        }
    }

    let mut all_els: std::collections::BTreeSet<String> = r_counts.keys().cloned().collect();
    all_els.extend(p_counts.keys().cloned());

    let mut delta_counts: BTreeMap<String, isize> = BTreeMap::new();
    for el in all_els {
        let r = r_counts.get(&el).copied().unwrap_or(0) as isize;
        let p = p_counts.get(&el).copied().unwrap_or(0) as isize;
        if p != r {
            delta_counts.insert(el, p - r);
        }
    }

    let balanced = delta_counts.is_empty();
    let summary = if balanced {
        "Balanced".to_string()
    } else {
        let mut gained = ElementCounts::new();
        let mut lost = ElementCounts::new();
        for (el, &d) in &delta_counts {
            if d > 0 {
                gained.insert(el.clone(), d as usize);
            } else {
                lost.insert(el.clone(), (-d) as usize);
            }
        }
        let mut parts = Vec::new();
        if !gained.is_empty() {
            parts.push(format!("+{}", hill_formula(&gained)));
        }
        if !lost.is_empty() {
            parts.push(format!("-{}", hill_formula(&lost)));
        }
        parts.join(" ")
    };

    AtomBalanceReport {
        balanced,
        reactant_counts: r_counts,
        product_counts: p_counts,
        delta_counts,
        summary,
    }
}

// ============================================================================
// Laboratory Stoichiometry Table Engine (`stoich.ts`)
// ============================================================================

/// Editable field kinds in a laboratory stoichiometry table (`Field` in `stoich.ts`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StoichField {
    Mass,
    Moles,
    Equiv,
    Volume,
    Density,
    Molarity,
    Purity,
    Measured,
}

/// User-entered amounts for one reactant, reagent, or product row/column (in base units:
/// `g` for mass, `mol` for moles, `mL` for volume, `g/mL` for density, `mol/L` for molarity,
/// `0.0..=1.0` fraction for purity).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StoichTypedInput {
    /// Mass in grams (`g`).
    pub mass_g: Option<f64>,
    /// Amount of substance in moles (`mol`).
    pub moles: Option<f64>,
    /// Molar equivalents relative to the limiting reagent (`1.0` = stoichiometric).
    pub equiv: Option<f64>,
    /// Liquid or solution volume in milliliters (`mL`).
    pub volume_ml: Option<f64>,
    /// Neat liquid density in `g/mL`.
    pub density_g_ml: Option<f64>,
    /// Solution concentration (molarity) in `mol/L` (`M`).
    pub molarity_m: Option<f64>,
    /// Reagent/product purity fraction (`0.0..=1.0`, default `1.0` = `100%`).
    pub purity_fraction: Option<f64>,
    /// Actual isolated product mass in grams (`g`) for computing `% yield`.
    pub measured_mass_g: Option<f64>,
}

impl StoichTypedInput {
    pub fn with_mass_mg(mut self, mg: f64) -> Self {
        self.mass_g = Some(mg * 1e-3);
        self
    }

    pub fn with_mass_g(mut self, g: f64) -> Self {
        self.mass_g = Some(g);
        self
    }

    pub fn with_mmol(mut self, mmol: f64) -> Self {
        self.moles = Some(mmol * 1e-3);
        self
    }

    pub fn with_equiv(mut self, eq: f64) -> Self {
        self.equiv = Some(eq);
        self
    }

    pub fn with_volume_ul(mut self, ul: f64) -> Self {
        self.volume_ml = Some(ul * 1e-3);
        self
    }

    pub fn with_volume_ml(mut self, ml: f64) -> Self {
        self.volume_ml = Some(ml);
        self
    }

    pub fn with_density(mut self, g_per_ml: f64) -> Self {
        self.density_g_ml = Some(g_per_ml);
        self
    }

    pub fn with_molarity(mut self, molar: f64) -> Self {
        self.molarity_m = Some(molar);
        self
    }

    pub fn with_purity_percent(mut self, pct: f64) -> Self {
        self.purity_fraction = Some(pct * 0.01);
        self
    }

    pub fn with_actual_mass_mg(mut self, mg: f64) -> Self {
        self.measured_mass_g = Some(mg * 1e-3);
        self
    }

    pub fn with_actual_mass_g(mut self, g: f64) -> Self {
        self.measured_mass_g = Some(g);
        self
    }
}

/// Component entry in a laboratory stoichiometry calculation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoichComponent {
    pub key: String,
    pub name: String,
    pub formula: String,
    pub mw: f64,
    pub is_product: bool,
    /// True if this reactant participates in limiting-reagent auto-detection (set `false` for pure solvents/catalysts when desired).
    pub can_be_limiting: bool,
    /// Balanced reaction stoichiometric coefficient (`1.0` by default).
    pub stoich_coeff: f64,
    pub typed: StoichTypedInput,
}

impl StoichComponent {
    /// Construct a stoichiometry component directly from an `AnalytMol`.
    pub fn from_mol(key: &str, name: &str, mol: &AnalytMol, is_product: bool) -> Self {
        let (cts, _) = element_counts_of_mol(mol);
        let formula = hill_formula(&cts);
        let mw = molecular_weight(&cts).unwrap_or(0.0);
        Self {
            key: key.to_string(),
            name: name.to_string(),
            formula,
            mw,
            is_product,
            can_be_limiting: !is_product,
            stoich_coeff: 1.0,
            typed: StoichTypedInput::default(),
        }
    }

    /// Construct a stoichiometry component from a reagent formula string (`"NaBH4"`, `"CuSO4·5H2O"`, `"K2CO3"`).
    pub fn from_formula(key: &str, formula_text: &str, is_product: bool) -> Option<Self> {
        let mw = formula_molar_mass(formula_text)?;
        Some(Self {
            key: key.to_string(),
            name: formula_text.to_string(),
            formula: formula_text.to_string(),
            mw,
            is_product,
            can_be_limiting: !is_product,
            stoich_coeff: 1.0,
            typed: StoichTypedInput::default(),
        })
    }
}

/// Solved stoichiometry row with complete laboratory units (`mg`/`g`, `mmol`/`mol`, `μL`/`mL`,
/// equivalents, limiting flag, theoretical yield, actual & `% yield`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoichSolvedRow {
    pub key: String,
    pub name: String,
    pub formula: String,
    pub mw: f64,
    pub is_product: bool,
    pub is_limiting: bool,
    pub purity_fraction: f64,
    pub density_g_ml: Option<f64>,
    pub molarity_m: Option<f64>,
    pub equiv: Option<f64>,
    /// Moles (`mol`) and millimoles (`mmol`) for reactants/reagents.
    pub moles: Option<f64>,
    pub mmol: Option<f64>,
    /// Bulk mass to weigh (`g` and `mg`, accounting for `purity_fraction`).
    pub mass_g: Option<f64>,
    pub mass_mg: Option<f64>,
    /// Liquid/solution volume to dispense (`mL` and `μL`).
    pub volume_ml: Option<f64>,
    pub volume_ul: Option<f64>,
    /// Product theoretical yield (`mol`, `mmol`, `g`, `mg`).
    pub theoretical_moles: Option<f64>,
    pub theoretical_mmol: Option<f64>,
    pub theoretical_mass_g: Option<f64>,
    pub theoretical_mass_mg: Option<f64>,
    /// Product actual isolated amount (`g`, `mg`, `mol`, `mmol`) and `% yield` (`0..=100+`).
    pub actual_mass_g: Option<f64>,
    pub actual_mass_mg: Option<f64>,
    pub actual_moles: Option<f64>,
    pub actual_mmol: Option<f64>,
    pub yield_fraction: Option<f64>,
    pub percent_yield: Option<f64>,
}

/// Solved laboratory stoichiometry table across all reactants, reagents, and products.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoichTableResult {
    /// Key of the limiting reagent (if determined).
    pub limiting_key: Option<String>,
    /// Reaction scale in moles (`limiting_moles / limiting_stoich_coeff`).
    pub reaction_scale_moles: Option<f64>,
    pub rows: Vec<StoichSolvedRow>,
}

impl StoichTableResult {
    pub fn row(&self, key: &str) -> Option<&StoichSolvedRow> {
        self.rows.iter().find(|r| r.key == key)
    }
}

fn own_moles_of(c: &StoichComponent) -> Option<f64> {
    let t = &c.typed;
    let purity = t.purity_fraction.unwrap_or(1.0).clamp(1e-6, 1.0);
    if let Some(m) = t.moles {
        return Some(m);
    }
    if let Some(mass_g) = t.mass_g {
        if c.mw > 0.0 {
            return Some((mass_g * purity) / c.mw);
        }
    }
    if let (Some(vol_ml), Some(dens)) = (t.volume_ml, t.density_g_ml) {
        if c.mw > 0.0 {
            return Some((vol_ml * dens * purity) / c.mw);
        }
    }
    if let (Some(vol_ml), Some(molar)) = (t.volume_ml, t.molarity_m) {
        return Some((vol_ml / 1000.0) * molar);
    }
    None
}

/// Solve a complete laboratory stoichiometry table (`solve` in `stoich.ts` enhanced with
/// multi-reactant limiting-reagent auto-detection when `explicit_limiting == None`).
pub fn solve_stoichiometry(
    components: &[StoichComponent],
    explicit_limiting: Option<&str>,
) -> StoichTableResult {
    let reactants: Vec<&StoichComponent> = components.iter().filter(|c| !c.is_product).collect();

    // Determine limiting reagent and 1.0-eq reaction scale (`scale_moles = n_lim / coeff_lim`)
    let mut limiting_key: Option<String> = None;
    let mut scale_moles: Option<f64> = None;

    if let Some(lim_k) = explicit_limiting {
        if let Some(lim_c) = reactants.iter().copied().find(|c| c.key == lim_k) {
            limiting_key = Some(lim_c.key.clone());
            let coeff = lim_c.stoich_coeff.max(1e-9);
            scale_moles = own_moles_of(lim_c).map(|m| m / coeff);
        }
    } else {
        // Auto-detect limiting reagent among `can_be_limiting` reactants that have fixed amounts entered:
        // choose the reactant with the smallest `own_moles / stoich_coeff`!
        let mut best: Option<(&StoichComponent, f64)> = None;
        for &rc in &reactants {
            if !rc.can_be_limiting {
                continue;
            }
            if let Some(m) = own_moles_of(rc) {
                let scale = m / rc.stoich_coeff.max(1e-9);
                if best.is_none_or(|(_, b_sc)| scale < b_sc - 1e-15) {
                    best = Some((rc, scale));
                }
            }
        }
        if let Some((lim_c, sc)) = best {
            limiting_key = Some(lim_c.key.clone());
            scale_moles = Some(sc);
        } else if let Some(&first_r) = reactants.first() {
            limiting_key = Some(first_r.key.clone());
        }
    }

    // Fallback: if limiting reactant's own amount wasn't typed, infer `scale_moles` from another reactant with both amount and `equiv`
    if scale_moles.is_none() {
        for &rc in &reactants {
            if let (Some(m), Some(eq)) = (own_moles_of(rc), rc.typed.equiv) {
                if eq > 0.0 {
                    scale_moles = Some(m / eq);
                    break;
                }
            }
        }
    }

    let mut rows = Vec::with_capacity(components.len());

    for c in components {
        let t = &c.typed;
        let purity = t.purity_fraction.unwrap_or(1.0).clamp(1e-6, 1.0);
        let coeff = c.stoich_coeff.max(1e-9);

        if !c.is_product {
            let is_lim = limiting_key.as_deref() == Some(c.key.as_str());
            let mut moles = own_moles_of(c);
            if moles.is_none() {
                if let Some(sc) = scale_moles {
                    let eq = if is_lim {
                        coeff
                    } else {
                        t.equiv.unwrap_or(coeff)
                    };
                    moles = Some(sc * eq);
                }
            }
            let mass_g = match moles {
                Some(m) if c.mw > 0.0 => Some((m * c.mw) / purity),
                _ => t.mass_g,
            };
            let volume_ml = t.volume_ml.or_else(|| {
                if let (Some(mg), Some(dens)) = (mass_g, t.density_g_ml) {
                    if dens > 0.0 {
                        return Some(mg / dens);
                    }
                }
                if let (Some(m), Some(molar)) = (moles, t.molarity_m) {
                    if molar > 0.0 {
                        return Some((m / molar) * 1000.0);
                    }
                }
                None
            });
            let equiv = match (moles, scale_moles) {
                (Some(m), Some(sc)) if sc > 0.0 => Some(m / sc),
                _ => t.equiv,
            };

            rows.push(StoichSolvedRow {
                key: c.key.clone(),
                name: c.name.clone(),
                formula: c.formula.clone(),
                mw: c.mw,
                is_product: false,
                is_limiting: is_lim,
                purity_fraction: purity,
                density_g_ml: t.density_g_ml,
                molarity_m: t.molarity_m,
                equiv,
                moles,
                mmol: moles.map(|m| m * 1e3),
                mass_g,
                mass_mg: mass_g.map(|g| g * 1e3),
                volume_ml,
                volume_ul: volume_ml.map(|v| v * 1e3),
                theoretical_moles: None,
                theoretical_mmol: None,
                theoretical_mass_g: None,
                theoretical_mass_mg: None,
                actual_mass_g: None,
                actual_mass_mg: None,
                actual_moles: None,
                actual_mmol: None,
                yield_fraction: None,
                percent_yield: None,
            });
        } else {
            let prod_equiv = t.equiv.unwrap_or(coeff);
            let theo_moles = scale_moles.map(|sc| sc * prod_equiv);
            let theo_mass_g = theo_moles.map(|m| m * c.mw);
            let actual_mass_g = t.measured_mass_g;
            let actual_moles = match actual_mass_g {
                Some(meas) if c.mw > 0.0 => Some((meas * purity) / c.mw),
                _ => None,
            };
            let yield_fraction = match (actual_moles, theo_moles) {
                (Some(am), Some(tm)) if tm > 0.0 => Some(am / tm),
                _ => None,
            };

            rows.push(StoichSolvedRow {
                key: c.key.clone(),
                name: c.name.clone(),
                formula: c.formula.clone(),
                mw: c.mw,
                is_product: true,
                is_limiting: false,
                purity_fraction: purity,
                density_g_ml: t.density_g_ml,
                molarity_m: t.molarity_m,
                equiv: Some(prod_equiv),
                moles: None,
                mmol: None,
                mass_g: None,
                mass_mg: None,
                volume_ml: None,
                volume_ul: None,
                theoretical_moles: theo_moles,
                theoretical_mmol: theo_moles.map(|m| m * 1e3),
                theoretical_mass_g: theo_mass_g,
                theoretical_mass_mg: theo_mass_g.map(|g| g * 1e3),
                actual_mass_g,
                actual_mass_mg: actual_mass_g.map(|g| g * 1e3),
                actual_moles,
                actual_mmol: actual_moles.map(|m| m * 1e3),
                yield_fraction,
                percent_yield: yield_fraction.map(|y| y * 100.0),
            });
        }
    }

    StoichTableResult {
        limiting_key,
        reaction_scale_moles: scale_moles,
        rows,
    }
}

/// Parse a user-typed laboratory amount (`"250 mg"`, `"1.2 eq"`, `"0.5 mL"`, `"500 uL"`, `"98%"`)
/// into base units (`g`, `mol`, `mL`, `g/mL`, `M`, `eq`, fraction), matching `parseAmount` in `stoich.ts`.
pub fn parse_stoich_amount(text: &str, field: StoichField) -> Option<f64> {
    let norm = text.trim().replace(',', ".").replace('μ', "µ");
    if norm.is_empty() {
        return None;
    }
    // Split leading numeric part (with optional scientific exponent) from unit suffix
    let bytes = norm.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() && (bytes[i].is_ascii_digit() || bytes[i] == b'.') {
        i += 1;
    }
    if i == 0 {
        return None;
    }
    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        let save = i;
        i += 1;
        if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
            i += 1;
        }
        let exp_start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == exp_start {
            i = save;
        }
    }
    let val = norm[..i].parse::<f64>().ok()?;
    if !val.is_finite() || val < 0.0 {
        return None;
    }
    let unit = norm[i..].trim();

    let factor: f64 = match field {
        StoichField::Mass | StoichField::Measured => match unit {
            "kg" => 1000.0,
            "g" => 1.0,
            "" | "mg" => 1e-3,
            "µg" | "ug" => 1e-6,
            _ => return None,
        },
        StoichField::Moles => match unit {
            "mol" => 1.0,
            "" | "mmol" => 1e-3,
            "µmol" | "umol" => 1e-6,
            "nmol" => 1e-9,
            _ => return None,
        },
        StoichField::Volume => match unit {
            "L" | "l" => 1000.0,
            "" | "mL" | "ml" => 1.0,
            "µL" | "uL" | "ul" => 1e-3,
            _ => return None,
        },
        StoichField::Density => match unit {
            "" | "g/mL" | "g/ml" | "g/cm3" | "kg/L" => 1.0,
            _ => return None,
        },
        StoichField::Molarity => match unit {
            "" | "M" | "mol/L" => 1.0,
            "mM" => 1e-3,
            _ => return None,
        },
        StoichField::Equiv => match unit {
            "" | "eq" | "equiv" => 1.0,
            _ => return None,
        },
        StoichField::Purity => match unit {
            "" | "%" => 0.01,
            _ => return None,
        },
    };

    let res = val * factor;
    if res.is_finite() {
        Some(res)
    } else {
        None
    }
}

/// Format a base-unit amount in the most readable laboratory unit (`"250 mg"`, `"1.25 mmol"`, `"500 µL"`, `"88.5%"`),
/// matching `formatAmount` in `stoich.ts`.
pub fn format_stoich_amount(value: f64, field: StoichField) -> String {
    if !value.is_finite() {
        return String::new();
    }
    let sig = |v: f64| -> String {
        let av = v.abs();
        if av >= 100.0 {
            format!("{v:.0}")
        } else if av >= 10.0 {
            format!("{v:.1}")
        } else {
            format!("{v:.2}")
        }
    };

    match field {
        StoichField::Purity => format!("{}%", sig(value * 100.0)),
        StoichField::Equiv => format!("{value:.2}"),
        StoichField::Density => format!("{} g/mL", sig(value)),
        StoichField::Molarity => format!("{} M", sig(value)),
        StoichField::Mass | StoichField::Measured => {
            let ladder = [("kg", 1000.0), ("g", 1.0), ("mg", 1e-3), ("µg", 1e-6)];
            let &(u, k) = ladder
                .iter()
                .find(|&&(_, f)| value >= f)
                .unwrap_or(&ladder[ladder.len() - 1]);
            format!("{} {u}", sig(value / k))
        }
        StoichField::Moles => {
            let ladder = [("mol", 1.0), ("mmol", 1e-3), ("µmol", 1e-6), ("nmol", 1e-9)];
            let &(u, k) = ladder
                .iter()
                .find(|&&(_, f)| value >= f)
                .unwrap_or(&ladder[ladder.len() - 1]);
            format!("{} {u}", sig(value / k))
        }
        StoichField::Volume => {
            let ladder = [("L", 1000.0), ("mL", 1.0), ("µL", 1e-3)];
            let &(u, k) = ladder
                .iter()
                .find(|&&(_, f)| value >= f)
                .unwrap_or(&ladder[ladder.len() - 1]);
            format!("{} {u}", sig(value / k))
        }
    }
}

/// Compute molar mass of a written reagent formula supporting parentheses, Unicode subscripts,
/// and hydrate/adduct dots (`"NaBH4"`, `"Ca(OH)2"`, `"CuSO4·5H2O"`), matching `formulaWeight` in `stoich.ts`.
pub fn formula_molar_mass(text: &str) -> Option<f64> {
    let norm: String = text
        .chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| match c {
            '₀' => '0',
            '₁' => '1',
            '₂' => '2',
            '₃' => '3',
            '₄' => '4',
            '₅' => '5',
            '₆' => '6',
            '₇' => '7',
            '₈' => '8',
            '₉' => '9',
            '·' | '•' => '.',
            other => other,
        })
        .collect();
    if norm.is_empty() {
        return None;
    }

    fn parse_group(s: &str, i: &mut usize, in_dot_part: bool) -> Option<f64> {
        let bytes = s.as_bytes();
        let mut total = 0.0;
        while *i < bytes.len() && bytes[*i] != b')' && !(in_dot_part && bytes[*i] == b'.') {
            if bytes[*i] == b'.' {
                *i += 1;
                let k_start = *i;
                while *i < bytes.len() && bytes[*i].is_ascii_digit() {
                    *i += 1;
                }
                let mult = if *i > k_start {
                    s[k_start..*i].parse::<f64>().ok()?
                } else {
                    1.0
                };
                let w = parse_group(s, i, true)?;
                total += mult * w;
                continue;
            }
            let w = if bytes[*i] == b'(' {
                *i += 1;
                let inner_w = parse_group(s, i, false)?;
                if *i >= bytes.len() || bytes[*i] != b')' {
                    return None;
                }
                *i += 1;
                inner_w
            } else if bytes[*i].is_ascii_uppercase() {
                let start = *i;
                *i += 1;
                if *i < bytes.len() && bytes[*i].is_ascii_lowercase() {
                    *i += 1;
                }
                let el = &s[start..*i];
                let cts = parse_formula_counts(el).ok()?;
                molecular_weight(&cts)?
            } else {
                return None;
            };

            let n_start = *i;
            while *i < bytes.len() && bytes[*i].is_ascii_digit() {
                *i += 1;
            }
            let count = if *i > n_start {
                s[n_start..*i].parse::<f64>().ok()?
            } else {
                1.0
            };
            total += w * count;
        }
        Some(total)
    }

    let mut pos = 0usize;
    let w = parse_group(&norm, &mut pos, false)?;
    if pos == norm.len() && w > 0.0 {
        Some(w)
    } else {
        None
    }
}
