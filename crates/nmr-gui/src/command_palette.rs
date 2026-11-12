//! Modal `Cmd/Ctrl+K` fuzzy command palette across all registered workbench actions (`T09.3`).
//!
//! Provenance: Original clean-room human-computer interaction design and $O(|Q|\cdot|T|)$
//! stack-allocated dynamic-programming subsequence / word-boundary / camelCase / acronym scorer
//! (`docs/specs/12-workbench-ux.md` §2). Zero external dependencies, zero heap allocations inside
//! the core character-alignment kernel, and full native + `wasm32-unknown-unknown` compatibility.

use eframe::egui;

use crate::workflow_bar::{ActionMetadata, WorkbenchAction};

const MAX_QUERY_CHARS: usize = 32;
const MAX_TEXT_CHARS: usize = 160;
const NEG_INF: i32 = i32::MIN / 4;

/// A registered command entry indexed by [`CommandPalette`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PaletteCommand {
    pub action: WorkbenchAction,
    pub category: &'static str,
    pub menu_label: &'static str,
    pub title: &'static str,
    pub description: &'static str,
    pub modifier_shortcut: Option<&'static str>,
    pub chord_badge: Option<&'static str>,
}

impl From<ActionMetadata> for PaletteCommand {
    fn from(meta: ActionMetadata) -> Self {
        Self {
            action: meta.action,
            category: meta.category,
            menu_label: meta.menu.label(),
            title: meta.title,
            description: meta.description,
            modifier_shortcut: meta.modifier_shortcut,
            chord_badge: meta.chord_badge,
        }
    }
}

/// Ranked command match returned by [`CommandPalette::search`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoredCommand {
    pub command: PaletteCommand,
    pub score: i32,
    /// Matched character indices (`0..title.chars().count()`) in `command.title` for UI highlighting.
    pub title_match_indices: Vec<usize>,
}

/// Normalize Unicode NMR superscripts/subscripts and Greek symbols to searchable ASCII equivalents.
#[inline]
fn fold_nmr_char(ch: char) -> char {
    match ch {
        '¹' => '1',
        '²' => '2',
        '³' => '3',
        '⁴' => '4',
        '⁵' => '5',
        '⁶' => '6',
        '⁷' => '7',
        '⁸' => '8',
        '⁹' => '9',
        '⁰' => '0',
        '₁' => '1',
        '₂' => '2',
        '₃' => '3',
        'α' => 'a',
        'β' => 'b',
        'γ' => 'g',
        'δ' | 'Δ' => 'd',
        'τ' => 't',
        'ρ' => 'r',
        'φ' => 'p',
        'ψ' => 'p',
        _ => ch.to_ascii_lowercase(),
    }
}

#[inline]
fn boundary_bonus_at(chars: &[char], idx: usize) -> i32 {
    if idx == 0 {
        return 44;
    }
    let prev = chars[idx - 1];
    let curr = chars[idx];
    if !prev.is_alphanumeric() {
        38
    } else if prev.is_ascii_lowercase() && curr.is_ascii_uppercase() {
        32
    } else if (prev.is_alphabetic() && curr.is_ascii_digit())
        || (prev.is_ascii_digit() && curr.is_alphabetic())
    {
        24
    } else {
        0
    }
}

/// Stack-allocated $O(|Q|\cdot|T|)$ subsequence + word-boundary + camel/acronym fuzzy scorer.
///
/// If `capture_positions` is true, reconstructs the exact optimal character indices in `text`.
pub fn fuzzy_match_field(
    query_token: &str,
    text: &str,
    capture_positions: bool,
) -> Option<(i32, Vec<usize>)> {
    if query_token.is_empty() {
        return Some((0, Vec::new()));
    }

    let mut q_buf = ['\0'; MAX_QUERY_CHARS];
    let mut q_len = 0usize;
    for ch in query_token.chars() {
        if q_len >= MAX_QUERY_CHARS {
            break;
        }
        q_buf[q_len] = fold_nmr_char(ch);
        q_len += 1;
    }
    if q_len == 0 {
        return Some((0, Vec::new()));
    }

    let mut t_orig = ['\0'; MAX_TEXT_CHARS];
    let mut t_fold = ['\0'; MAX_TEXT_CHARS];
    let mut t_bonus = [0i32; MAX_TEXT_CHARS];
    let mut t_len = 0usize;
    for ch in text.chars() {
        if t_len >= MAX_TEXT_CHARS {
            break;
        }
        t_orig[t_len] = ch;
        t_fold[t_len] = fold_nmr_char(ch);
        t_len += 1;
    }
    if q_len > t_len {
        return None;
    }

    for (j, slot) in t_bonus.iter_mut().enumerate().take(t_len) {
        *slot = boundary_bonus_at(&t_orig[..t_len], j);
    }

    let mut prev_dp = [NEG_INF; MAX_TEXT_CHARS];
    let mut curr_dp = [NEG_INF; MAX_TEXT_CHARS];
    let mut backptr = [[0u8; MAX_TEXT_CHARS]; MAX_QUERY_CHARS];

    // Row 0 (`i = 0`)
    let q0 = q_buf[0];
    let mut any_row = false;
    for j in 0..t_len {
        if t_fold[j] == q0 {
            prev_dp[j] = 18 + t_bonus[j] - (j as i32);
            backptr[0][j] = j as u8;
            any_row = true;
        }
    }
    if !any_row {
        return None;
    }

    // Rows `1..q_len`
    for i in 1..q_len {
        curr_dp.fill(NEG_INF);
        let qi = q_buf[i];
        let mut best_gapped_val = NEG_INF;
        let mut best_gapped_idx = 0usize;
        let mut row_has_match = false;

        for j in 1..t_len {
            // Update running max of `prev_dp[k] + 2 * k` for gapped transitions (`k <= j - 2`)
            if j >= 2 && prev_dp[j - 2] > NEG_INF {
                let candidate = prev_dp[j - 2] + 2 * ((j - 2) as i32);
                if candidate > best_gapped_val {
                    best_gapped_val = candidate;
                    best_gapped_idx = j - 2;
                }
            }

            if t_fold[j] != qi {
                continue;
            }

            let mut best_here = NEG_INF;
            let mut chosen_prev = 0usize;

            // 1. Consecutive character match (`k = j - 1`)
            if prev_dp[j - 1] > NEG_INF {
                let consec = prev_dp[j - 1] + 18 + 30 + (t_bonus[j] / 2);
                best_here = consec;
                chosen_prev = j - 1;
            }

            // 2. Gapped match (`k <= j - 2`) with `-2 * (j - k - 1)` gap cost
            if best_gapped_val > NEG_INF {
                let gapped = best_gapped_val - 2 * ((j - 1) as i32) + 18 + t_bonus[j];
                if gapped > best_here {
                    best_here = gapped;
                    chosen_prev = best_gapped_idx;
                }
            }

            if best_here > NEG_INF {
                curr_dp[j] = best_here;
                backptr[i][j] = chosen_prev as u8;
                row_has_match = true;
            }
        }

        if !row_has_match {
            return None;
        }
        prev_dp = curr_dp;
    }

    let mut best_end = 0usize;
    let mut best_score = NEG_INF;
    for (j, &sc) in prev_dp[..t_len].iter().enumerate() {
        if sc > best_score {
            best_score = sc;
            best_end = j;
        }
    }
    if best_score <= NEG_INF / 2 {
        return None;
    }

    // Exact prefix bonus when query matches from char 0 with no gaps
    if best_end + 1 == q_len {
        best_score += 28;
    }

    if !capture_positions {
        return Some((best_score, Vec::new()));
    }

    let mut indices = vec![0usize; q_len];
    let mut cursor = best_end;
    for i in (0..q_len).rev() {
        indices[i] = cursor;
        if i > 0 {
            cursor = backptr[i][cursor] as usize;
        }
    }
    Some((best_score, indices))
}

fn eq_ascii_icase(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

fn starts_with_ascii_icase(text: &str, prefix: &str) -> bool {
    text.len() >= prefix.len()
        && text
            .get(..prefix.len())
            .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
}

/// Modal `Cmd/Ctrl+K` fuzzy command palette state and renderer.
#[derive(Debug, Clone)]
pub struct CommandPalette {
    commands: Vec<PaletteCommand>,
    pub open: bool,
    pub query: String,
    pub selected_index: usize,
    focus_input_next_frame: bool,
}

impl Default for CommandPalette {
    fn default() -> Self {
        Self::new()
    }
}

impl CommandPalette {
    /// Construct a palette pre-populated with all [`WorkbenchAction::ALL`] commands (`>= 55`).
    pub fn new() -> Self {
        let commands = WorkbenchAction::ALL
            .iter()
            .copied()
            .map(|a| PaletteCommand::from(a.metadata()))
            .collect();
        Self {
            commands,
            open: false,
            query: String::new(),
            selected_index: 0,
            focus_input_next_frame: false,
        }
    }

    /// Number of registered commands indexed by the palette.
    #[inline]
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Read-only slice of all registered [`PaletteCommand`] entries.
    #[inline]
    pub fn commands(&self) -> &[PaletteCommand] {
        &self.commands
    }

    /// Open the command palette modal, clearing the search query and resetting selection to `0`.
    pub fn open_palette(&mut self) {
        self.open = true;
        self.query.clear();
        self.selected_index = 0;
        self.focus_input_next_frame = true;
    }

    /// Close (`Esc`) the command palette without emitting an action.
    pub fn close_palette(&mut self) {
        self.open = false;
        self.focus_input_next_frame = false;
    }

    /// Toggle open/closed state (bound to `Cmd/Ctrl+K`).
    pub fn toggle(&mut self) {
        if self.open {
            self.close_palette();
        } else {
            self.open_palette();
        }
    }

    /// Update the active query string and clamp `selected_index`.
    pub fn set_query(&mut self, query: impl Into<String>) {
        self.query = query.into();
        self.selected_index = 0;
    }

    /// Move selection highlight down (`ArrowDown`), wrapping cleanly across `result_count`.
    pub fn move_selection_down(&mut self, result_count: usize) {
        if result_count == 0 {
            self.selected_index = 0;
        } else {
            self.selected_index = (self.selected_index + 1) % result_count;
        }
    }

    /// Move selection highlight up (`ArrowUp`), wrapping cleanly across `result_count`.
    pub fn move_selection_up(&mut self, result_count: usize) {
        if result_count == 0 {
            self.selected_index = 0;
        } else if self.selected_index == 0 {
            self.selected_index = result_count - 1;
        } else {
            self.selected_index -= 1;
        }
    }

    /// Confirm (`Enter`) the currently highlighted search result, closing the palette.
    pub fn confirm_selected(&mut self) -> Option<WorkbenchAction> {
        if !self.open {
            return None;
        }
        let results = self.search(&self.query);
        let picked = results
            .get(self.selected_index)
            .or_else(|| results.first())
            .map(|r| r.command.action);
        self.close_palette();
        picked
    }

    /// Score and rank all registered commands against `query`.
    ///
    /// Empty/whitespace queries return all commands in canonical menu order.
    pub fn search(&self, query: &str) -> Vec<ScoredCommand> {
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return self
                .commands
                .iter()
                .enumerate()
                .map(|(ord, &cmd)| ScoredCommand {
                    command: cmd,
                    score: 1000 - (ord as i32),
                    title_match_indices: Vec::new(),
                })
                .collect();
        }

        // Fast stack array of up to 6 whitespace-delimited query tokens (zero Vec alloc for tokens)
        let mut tokens = [""; 6];
        let mut n_tokens = 0usize;
        for tok in trimmed.split_whitespace() {
            if n_tokens < tokens.len() {
                tokens[n_tokens] = tok;
                n_tokens += 1;
            }
        }

        let mut ranked: Vec<(i32, usize, PaletteCommand)> = Vec::with_capacity(self.commands.len());

        'cmd_loop: for (ord, &cmd) in self.commands.iter().enumerate() {
            let mut total_score = 0i32;

            for &tok in &tokens[..n_tokens] {
                let mut best_tok_score = NEG_INF;

                // 1. Two-key chord exact or prefix match
                if let Some(chord) = cmd.chord_badge {
                    if eq_ascii_icase(chord, tok) {
                        best_tok_score = best_tok_score.max(650);
                    } else if starts_with_ascii_icase(chord, tok) {
                        best_tok_score = best_tok_score.max(240);
                    }
                }

                // 2. Modifier shortcut match
                if let Some(sc) = cmd.modifier_shortcut {
                    if let Some((s, _)) = fuzzy_match_field(tok, sc, false) {
                        best_tok_score = best_tok_score.max(s * 2 + 40);
                    }
                }

                // 3. Title match (highest prose weight)
                if let Some((s, _)) = fuzzy_match_field(tok, cmd.title, false) {
                    best_tok_score = best_tok_score.max(s * 3 + 60);
                }

                // 4. Category / Menu match
                if let Some((s, _)) = fuzzy_match_field(tok, cmd.category, false) {
                    best_tok_score = best_tok_score.max(s * 2 + 10);
                } else if let Some((s, _)) = fuzzy_match_field(tok, cmd.menu_label, false) {
                    best_tok_score = best_tok_score.max(s * 2 + 10);
                }

                // 5. Description match (only checked if higher-priority fields scored modestly)
                if best_tok_score < 160 {
                    if let Some((s, _)) = fuzzy_match_field(tok, cmd.description, false) {
                        best_tok_score = best_tok_score.max(s);
                    }
                }

                if best_tok_score <= NEG_INF / 2 {
                    continue 'cmd_loop;
                }
                total_score += best_tok_score;
            }

            ranked.push((total_score, ord, cmd));
        }

        ranked.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));

        // Reconstruct highlight indices only for the top 20 visible items to keep latency ultra-low
        ranked
            .into_iter()
            .enumerate()
            .map(|(rank_idx, (score, _, cmd))| {
                let title_match_indices = if rank_idx < 20 && n_tokens > 0 {
                    collect_title_highlight_indices(&tokens[..n_tokens], cmd.title)
                } else {
                    Vec::new()
                };
                ScoredCommand {
                    command: cmd,
                    score,
                    title_match_indices,
                }
            })
            .collect()
    }

    /// Poll `egui` input for `Cmd/Ctrl+K` and toggle the palette if pressed.
    pub fn handle_global_shortcut(&mut self, ctx: &egui::Context) -> bool {
        let toggled = ctx.input_mut(|i| {
            let cmd_or_ctrl = i.modifiers.command || i.modifiers.ctrl;
            if cmd_or_ctrl && !i.modifiers.alt && i.key_pressed(egui::Key::K) {
                i.consume_key(i.modifiers, egui::Key::K);
                true
            } else {
                false
            }
        });
        if toggled {
            self.toggle();
        }
        toggled
    }

    /// Render the modal `Cmd/Ctrl+K` command palette window and return any selected action.
    pub fn show(&mut self, ctx: &egui::Context) -> Option<WorkbenchAction> {
        if !self.open {
            return None;
        }

        let mut results = self.search(&self.query);
        if self.selected_index >= results.len() {
            self.selected_index = results.len().saturating_sub(1);
        }

        // Keyboard navigation (`Up` / `Down` / `Enter` / `Esc`) while modal is open
        let (nav_up, nav_down, nav_enter, nav_esc) = ctx.input_mut(|i| {
            let up = i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp);
            let down = i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown);
            let enter = i.consume_key(egui::Modifiers::NONE, egui::Key::Enter);
            let esc = i.consume_key(egui::Modifiers::NONE, egui::Key::Escape);
            (up, down, enter, esc)
        });

        if nav_esc {
            self.close_palette();
            return None;
        }
        if nav_up {
            self.move_selection_up(results.len());
        }
        if nav_down {
            self.move_selection_down(results.len());
        }
        if nav_enter {
            let chosen = results.get(self.selected_index).map(|r| r.command.action);
            self.close_palette();
            return chosen;
        }

        let mut chosen_action: Option<WorkbenchAction> = None;
        let mut window_open = self.open;

        egui::Window::new("⌘ Command Palette")
            .id(egui::Id::unique("nmr_workbench_command_palette"))
            .open(&mut window_open)
            .collapsible(false)
            .resizable(false)
            .anchor(egui::Align2::CENTER_TOP, egui::vec2(0.0, 56.0))
            .fixed_size(egui::vec2(660.0, 430.0))
            .show(ctx, |ui| {
                let prev_query = self.query.clone();
                let edit_resp = ui.add(
                    egui::TextEdit::singleline(&mut self.query)
                        .hint_text(
                            "Type a command, workflow step, nucleus, or two-key chord (e.g. vt, pa, st, nef, cest, rba)…",
                        )
                        .desired_width(f32::INFINITY),
                );

                if self.focus_input_next_frame {
                    edit_resp.request_focus();
                    self.focus_input_next_frame = false;
                }
                if self.query != prev_query {
                    self.selected_index = 0;
                    results = self.search(&self.query);
                }

                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(format!(
                            "{} matching command(s) of {}",
                            results.len(),
                            self.commands.len()
                        ))
                        .small()
                        .weak(),
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(
                            egui::RichText::new("↑/↓ Navigate   ↵ Run   Esc Close")
                                .small()
                                .monospace()
                                .weak(),
                        );
                    });
                });
                ui.separator();

                if results.is_empty() {
                    ui.vertical_centered(|ui| {
                        ui.add_space(28.0);
                        ui.label(
                            egui::RichText::new("No matching workbench commands found.")
                                .color(egui::Color32::from_rgb(165, 180, 205)),
                        );
                    });
                    return;
                }

                egui::ScrollArea::vertical()
                    .id_salt("command_palette_results_scroll")
                    .max_height(340.0)
                    .show(ui, |ui| {
                        for (idx, item) in results.iter().enumerate() {
                            let is_sel = idx == self.selected_index;
                            let row_fill = if is_sel {
                                egui::Color32::from_rgb(34, 62, 96)
                            } else {
                                egui::Color32::from_rgb(22, 28, 39)
                            };
                            let row_stroke = if is_sel {
                                egui::Stroke::new(1.2, egui::Color32::from_rgb(98, 205, 255))
                            } else {
                                egui::Stroke::new(0.6, egui::Color32::from_rgb(42, 52, 68))
                            };

                            let frame_resp = egui::Frame::group(ui.style())
                                .fill(row_fill)
                                .stroke(row_stroke)
                                .inner_margin(egui::Margin::symmetric(8, 5))
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.label(
                                            egui::RichText::new(format!(
                                                "[{}]",
                                                item.command.category
                                            ))
                                            .monospace()
                                            .size(10.5)
                                            .color(egui::Color32::from_rgb(130, 195, 235)),
                                        );

                                        ui.add(egui::Label::new(build_highlighted_title_job(
                                            item.command.title,
                                            &item.title_match_indices,
                                        )));

                                        ui.with_layout(
                                            egui::Layout::right_to_left(egui::Align::Center),
                                            |ui| {
                                                if let Some(chord) = item.command.chord_badge {
                                                    draw_keycap_pill(
                                                        ui,
                                                        &format!("chord: {chord}"),
                                                        egui::Color32::from_rgb(28, 72, 62),
                                                        egui::Color32::from_rgb(135, 248, 205),
                                                    );
                                                }
                                                if let Some(sc) = item.command.modifier_shortcut {
                                                    draw_keycap_pill(
                                                        ui,
                                                        sc,
                                                        egui::Color32::from_rgb(38, 48, 68),
                                                        egui::Color32::from_rgb(195, 220, 255),
                                                    );
                                                }
                                            },
                                        );
                                    });

                                    ui.label(
                                        egui::RichText::new(item.command.description)
                                            .size(11.0)
                                            .color(egui::Color32::from_rgb(165, 180, 202)),
                                    );
                                });

                            let interact = ui.interact(
                                frame_resp.response.rect,
                                ui.make_persistent_id(("cmd_palette_row", idx)),
                                egui::Sense::click(),
                            );
                            if interact.hovered() {
                                self.selected_index = idx;
                            }
                            if interact.clicked() {
                                chosen_action = Some(item.command.action);
                            }
                            if is_sel && (nav_up || nav_down) {
                                frame_resp
                                    .response
                                    .scroll_to_me(Some(egui::Align::Center));
                            }
                        }
                    });
            });

        if chosen_action.is_some() {
            self.close_palette();
            return chosen_action;
        }
        if !window_open {
            self.close_palette();
        }
        None
    }
}

fn collect_title_highlight_indices(tokens: &[&str], title: &str) -> Vec<usize> {
    let mut all = Vec::new();
    for &tok in tokens {
        if let Some((_, indices)) = fuzzy_match_field(tok, title, true) {
            for i in indices {
                if !all.contains(&i) {
                    all.push(i);
                }
            }
        }
    }
    all.sort_unstable();
    all
}

fn build_highlighted_title_job(title: &str, matched: &[usize]) -> egui::text::LayoutJob {
    let mut job = egui::text::LayoutJob::default();
    let normal_fmt = egui::TextFormat {
        font_id: egui::FontId::proportional(13.0),
        color: egui::Color32::from_rgb(232, 240, 252),
        ..Default::default()
    };
    let hi_fmt = egui::TextFormat {
        font_id: egui::FontId::proportional(13.0),
        color: egui::Color32::from_rgb(105, 235, 255),
        underline: egui::Stroke::new(1.0, egui::Color32::from_rgb(80, 195, 235)),
        ..Default::default()
    };

    let mut buf = String::new();
    let mut cur_hi = false;
    for (char_idx, ch) in title.chars().enumerate() {
        let is_hi = matched.binary_search(&char_idx).is_ok();
        if char_idx > 0 && is_hi != cur_hi && !buf.is_empty() {
            job.append(
                &buf,
                0.0,
                if cur_hi {
                    hi_fmt.clone()
                } else {
                    normal_fmt.clone()
                },
            );
            buf.clear();
        }
        cur_hi = is_hi;
        buf.push(ch);
    }
    if !buf.is_empty() {
        job.append(&buf, 0.0, if cur_hi { hi_fmt } else { normal_fmt });
    }
    job
}

fn draw_keycap_pill(ui: &mut egui::Ui, label: &str, fill: egui::Color32, text_col: egui::Color32) {
    egui::Frame::NONE
        .fill(fill)
        .stroke(egui::Stroke::new(0.8, text_col.gamma_multiply(0.55)))
        .corner_radius(3.0)
        .inner_margin(egui::Margin::symmetric(5, 2))
        .show(ui, |ui| {
            ui.label(
                egui::RichText::new(label)
                    .monospace()
                    .size(10.5)
                    .strong()
                    .color(text_col),
            );
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_palette_registers_at_least_55_commands_with_shortcuts_and_chords() {
        let palette = CommandPalette::new();
        assert!(palette.len() >= 55);

        // Empty query returns every command
        let all = palette.search("");
        assert_eq!(all.len(), palette.len());

        // Verify both modifier shortcuts and two-key chord badges are represented
        let n_chords = palette
            .commands()
            .iter()
            .filter(|c| c.chord_badge.is_some())
            .count();
        let n_mods = palette
            .commands()
            .iter()
            .filter(|c| c.modifier_shortcut.is_some())
            .count();
        assert!(
            n_chords >= 14,
            "Expected >= 14 chord badges, got {n_chords}"
        );
        assert!(
            n_mods >= 6,
            "Expected >= 6 modifier shortcuts, got {n_mods}"
        );
    }

    #[test]
    fn test_fuzzy_scorer_subsequence_word_boundary_acronym_and_chords() {
        let palette = CommandPalette::new();

        // 1. Exact two-key chord lookup ranks that exact accelerator #1
        for (chord, expected) in [
            ("vt", WorkbenchAction::OpenContourThresholdPopup),
            ("pa", WorkbenchAction::AddPeakAtCrosshair),
            ("pd", WorkbenchAction::DeleteSelectedPeaks),
            ("pc", WorkbenchAction::SnapPeakParabolicCenter),
            ("pl", WorkbenchAction::TogglePeakLabels),
            ("oz", WorkbenchAction::ZoomFullSweep),
            ("ct", WorkbenchAction::RaiseContourFloor),
            ("cb", WorkbenchAction::LowerContourFloor),
            ("st", WorkbenchAction::TileSequentialStrips),
            ("at", WorkbenchAction::OpenResidueAssignmentPopover),
        ] {
            let hits = palette.search(chord);
            assert!(!hits.is_empty(), "No hits for chord {chord}");
            assert_eq!(
                hits[0].command.action, expected,
                "Chord query '{chord}' should rank {expected:?} #1, got {:?}",
                hits[0].command.action
            );
        }

        // 2. Acronym / word-boundary matching: "rba" -> "Run Automated Backbone Assignment" or backbone assignment
        let rba = palette.search("raba");
        assert!(!rba.is_empty());
        assert_eq!(
            rba[0].command.action,
            WorkbenchAction::RunBackboneAssignment
        );
        assert_eq!(rba[0].title_match_indices.len(), 4);

        // 3. Subsequence + domain multi-token queries
        let nus = palette.search("nus fista");
        assert!(!nus.is_empty());
        assert_eq!(
            nus[0].command.action,
            WorkbenchAction::ReconstructNusIstFista
        );

        let nef = palette.search("nef save");
        assert!(!nef.is_empty());
        assert!(matches!(
            nef[0].command.action,
            WorkbenchAction::SaveSessionNef | WorkbenchAction::SaveSessionNefAs
        ));

        let cest = palette.search("dynamics cest");
        assert!(!cest.is_empty());
        assert_eq!(cest[0].command.action, WorkbenchAction::AnalyzeCestExchange);
    }

    #[test]
    fn test_palette_keyboard_navigation_up_down_enter_esc() {
        let mut palette = CommandPalette::new();
        assert!(!palette.open);

        palette.open_palette();
        assert!(palette.open);
        assert_eq!(palette.selected_index, 0);

        palette.set_query("contour");
        let hits = palette.search(&palette.query);
        assert!(hits.len() >= 3);

        // Wrap-around up from 0 -> last
        palette.move_selection_up(hits.len());
        assert_eq!(palette.selected_index, hits.len() - 1);

        // Wrap-around down from last -> 0 -> 1
        palette.move_selection_down(hits.len());
        assert_eq!(palette.selected_index, 0);
        palette.move_selection_down(hits.len());
        assert_eq!(palette.selected_index, 1);

        let expected = hits[1].command.action;
        let confirmed = palette.confirm_selected();
        assert_eq!(confirmed, Some(expected));
        assert!(!palette.open);

        // Esc / close emits nothing
        palette.open_palette();
        palette.close_palette();
        assert!(!palette.open);
        assert_eq!(palette.confirm_selected(), None);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn test_fuzzy_search_latency_under_1ms_across_all_commands() {
        let palette = CommandPalette::new();
        assert!(palette.len() >= 55);

        let sample_queries = [
            "vt",
            "pa",
            "raba",
            "hncacb",
            "nef save",
            "parabolic snap",
            "lipari szabo",
            "octet svg",
            "qc abstain",
            "contour floor",
        ];

        // Warmup pass
        for q in sample_queries {
            let _ = palette.search(q);
        }

        for q in sample_queries {
            let t0 = std::time::Instant::now();
            let res = std::hint::black_box(palette.search(std::hint::black_box(q)));
            let elapsed = t0.elapsed();
            assert!(!res.is_empty(), "Query '{q}' returned empty results");
            assert!(
                elapsed.as_secs_f64() * 1000.0 < 1.0,
                "CommandPalette::search('{q}') took {:.4} ms (>= 1.0 ms gate)",
                elapsed.as_secs_f64() * 1000.0
            );
        }
    }
}
