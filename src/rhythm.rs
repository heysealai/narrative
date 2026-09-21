//! Rhythms: when the person is around, counted mechanically from activity
//! times in their zone. The pattern pass writes the result on the rhythms crown.

use chrono::{Datelike, TimeZone, Timelike};

use crate::clock::zone;
use crate::model::{age_str, Rhythm};

const DAY: u64 = 86_400;

pub const MIN_POINTS: usize = 20;
pub const MIN_SPAN_SECS: u64 = 7 * DAY;
pub const SESSION_GAP_SECS: u64 = 3_600;
pub const WINDOW_SHARE: f64 = 0.8;
pub const QUIET_SHARE: f64 = 0.4;
pub const REFRESH_SECS: u64 = 7 * DAY;
pub const REFRESH_POINTS: u32 = 20;
pub const IDLE_SECS: u64 = 14 * DAY;

const DAY_NAMES: [&str; 7] = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];

pub fn compute(points: &[u64], timezone: &str, now: u64) -> Option<Rhythm> {
    let tz = zone(timezone)?;
    let mut points: Vec<u64> = points.to_vec();
    points.sort_unstable();
    let (first, last) = (*points.first()?, *points.last()?);
    if points.len() < MIN_POINTS || last.saturating_sub(first) < MIN_SPAN_SECS {
        return None;
    }
    let mut hours = vec![0u32; 24];
    let mut days = vec![0u32; 7];
    let since = now.saturating_sub(30 * DAY);
    let mut dates_30d: Vec<i32> = Vec::new();
    let mut seen_hours: Vec<(i32, u32)> = Vec::new();
    let mut seen_dates: Vec<(i32, usize)> = Vec::new();
    for &t in &points {
        let local = tz.timestamp_opt(t as i64, 0).single()?;
        let date = local.num_days_from_ce();
        if !seen_hours.contains(&(date, local.hour())) {
            seen_hours.push((date, local.hour()));
            hours[local.hour() as usize] += 1;
        }
        let weekday = local.weekday().num_days_from_monday() as usize;
        if !seen_dates.contains(&(date, weekday)) {
            seen_dates.push((date, weekday));
            days[weekday] += 1;
        }
        if t >= since {
            dates_30d.push(date);
        }
    }
    dates_30d.sort_unstable();
    dates_30d.dedup();
    let (window_start, window_len) = densest_window(&hours);
    let sessions_30d = sessions(&points).into_iter().filter(|&(_, end)| end >= since).count() as u32;
    Some(Rhythm {
        timezone: timezone.to_string(),
        n: points.len() as u32,
        first,
        last,
        hours,
        days,
        window_start,
        window_len,
        active_days_30d: dates_30d.len() as u32,
        sessions_30d,
        computed_at: now,
    })
}

fn densest_window(hours: &[u32]) -> (u8, u8) {
    let total: u32 = hours.iter().sum();
    let need = (total as f64 * WINDOW_SHARE).ceil() as u32;
    for len in 1..=24u8 {
        let mut best: Option<(u8, u32)> = None;
        for start in 0..24u8 {
            let sum: u32 = (0..len).map(|i| hours[((start + i) % 24) as usize]).sum();
            if sum >= need && best.is_none_or(|(_, b)| sum > b) {
                best = Some((start, sum));
            }
        }
        if let Some((start, _)) = best {
            return (start, len);
        }
    }
    (0, 24)
}

fn sessions(points: &[u64]) -> Vec<(u64, u64)> {
    let mut out: Vec<(u64, u64)> = Vec::new();
    for &t in points {
        match out.last_mut() {
            Some((_, end)) if t.saturating_sub(*end) <= SESSION_GAP_SECS => *end = t,
            _ => out.push((t, t)),
        }
    }
    out
}

pub fn is_idle(r: &Rhythm, now: u64) -> bool {
    now.saturating_sub(r.last) > IDLE_SECS
}

fn period(window_start: u8, window_len: u8) -> &'static str {
    let mid = (window_start as u32 + window_len as u32 / 2) % 24;
    match mid {
        5..=11 => "in the mornings",
        12..=16 => "in the afternoons",
        17..=23 => "in the evenings",
        _ => "late at night",
    }
}

fn quiet_days(days: &[u32]) -> String {
    let total: u32 = days.iter().sum();
    let floor = total as f64 / 7.0 * QUIET_SHARE;
    let quiet: Vec<usize> = (0..7).filter(|&d| (days[d] as f64) < floor).collect();
    let weekend = quiet == [5, 6];
    let only_weekend = (0..5).all(|d| quiet.contains(&d)) && !quiet.contains(&5) && !quiet.contains(&6);
    match (quiet.len(), weekend, only_weekend) {
        (0, _, _) => "every day of the week".to_string(),
        (_, true, _) => "quiet on weekends".to_string(),
        (_, _, true) => "mostly on weekends".to_string(),
        (n, _, _) if n >= 4 => {
            let on: Vec<&str> = (0..7).filter(|d| !quiet.contains(d)).map(|d| DAY_NAMES[d]).collect();
            format!("mostly on {}", join_names(&on))
        }
        _ => {
            let names: Vec<&str> = quiet.iter().map(|&d| DAY_NAMES[d]).collect();
            format!("quiet on {}", join_names(&names))
        }
    }
}

fn join_names(names: &[&str]) -> String {
    match names {
        [] => String::new(),
        [one] => format!("{one}s"),
        [head @ .., tail] => format!("{} and {tail}s", head.iter().map(|n| format!("{n}s")).collect::<Vec<_>>().join(", ")),
    }
}

fn abbreviation(r: &Rhythm) -> String {
    zone(&r.timezone)
        .and_then(|tz| tz.timestamp_opt(r.last as i64, 0).single())
        .map(|t| t.format("%Z").to_string())
        .unwrap_or_else(|| r.timezone.clone())
}

pub fn render_line(r: &Rhythm, now: u64) -> String {
    let end = (r.window_start as u32 + r.window_len as u32) % 24;
    let hours = format!("{:02}:00–{:02}:00 {}", r.window_start, end, abbreviation(r));
    let when = match r.window_len >= 16 {
        true => format!("most of the day, {hours}, {}", quiet_days(&r.days)),
        false => format!("{}, usually {hours}, {}", period(r.window_start, r.window_len), quiet_days(&r.days)),
    };
    match is_idle(r, now) {
        true => format!("Was around {when}; nothing since {}.", age_str(now, r.last)),
        false => format!("Around {when}; active {} of the last 30 days.", r.active_days_30d),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NOW: u64 = 1_790_000_000;
    const HOUR: u64 = 3_600;

    fn evenings_in_tokyo(days: u64, skip_sunday: bool) -> Vec<u64> {
        let tz = zone("Asia/Tokyo").unwrap();
        let mut out = Vec::new();
        for d in 0..days {
            let t = NOW - d * DAY;
            let local = tz.timestamp_opt(t as i64, 0).single().unwrap();
            if skip_sunday && local.weekday().num_days_from_monday() == 6 {
                continue;
            }
            let midnight = local.date_naive().and_hms_opt(0, 0, 0).unwrap();
            let base = tz.from_local_datetime(&midnight).single().unwrap().timestamp() as u64;
            out.extend([base + 21 * HOUR, base + 22 * HOUR + 600, base + 23 * HOUR + 1_200]);
        }
        out.sort_unstable();
        out
    }

    #[test]
    fn evenings_in_one_zone_read_as_evenings_in_that_zone() {
        let points = evenings_in_tokyo(20, true);
        let r = compute(&points, "Asia/Tokyo", NOW).unwrap();
        assert_eq!((r.window_start, r.window_len), (21, 3));
        assert_eq!(r.days[6], 0, "no Sunday points");
        let active_days: u32 = r.days.iter().sum();
        assert_eq!(r.hours[21], active_days, "an hour counts once per day it was active, not once per mark");
        assert_eq!(r.hours.iter().sum::<u32>(), 3 * r.days.iter().sum::<u32>(), "three active hours on each active day");
        let line = render_line(&r, NOW);
        assert!(line.starts_with("Around in the evenings, usually 21:00–00:00 JST, quiet on Sundays; active "), "{line}");
        assert!(line.ends_with(" of the last 30 days."), "{line}");
        let utc = compute(&points, "UTC", NOW).unwrap();
        assert_eq!((utc.window_start, utc.window_len), (12, 3), "the same points nine hours earlier in UTC");
    }

    #[test]
    fn a_burst_of_marks_in_one_hour_counts_as_one_active_hour() {
        let mut points = evenings_in_tokyo(20, false);
        let burst_at = points[0];
        points.extend((1..200).map(|i| burst_at + i));
        let r = compute(&points, "Asia/Tokyo", NOW).unwrap();
        assert_eq!(r.n, 60 + 199);
        assert_eq!(r.hours.iter().sum::<u32>(), 60, "presence, not volume");
        assert_eq!((r.window_start, r.window_len), (21, 3));
    }

    #[test]
    fn too_little_activity_or_no_zone_yields_nothing() {
        let points = evenings_in_tokyo(20, false);
        assert!(compute(&points[..10], "Asia/Tokyo", NOW).is_none(), "under MIN_POINTS");
        assert!(compute(&points, "Mars/Olympus", NOW).is_none(), "unknown zone");
        let one_day: Vec<u64> = (0..30).map(|i| NOW - i * 60).collect();
        assert!(compute(&one_day, "UTC", NOW).is_none(), "under MIN_SPAN");
    }

    #[test]
    fn a_long_idle_rhythm_reads_in_the_past_tense() {
        let points = evenings_in_tokyo(20, false);
        let later = NOW + 20 * DAY;
        let r = compute(&points, "Asia/Tokyo", later).unwrap();
        assert!(is_idle(&r, later));
        let line = render_line(&r, later);
        assert!(line.starts_with("Was around in the evenings"), "{line}");
        assert!(line.ends_with("nothing since 19d ago."), "{line}");
        assert_eq!(r.active_days_30d, 11, "the thirty-day window reaches back to day ten of the twenty, inclusive");
    }

    #[test]
    fn sessions_split_on_an_hour_of_silence() {
        let points = [0, 600, 1_200, 6_000, 6_300];
        assert_eq!(sessions(&points), vec![(0, 1_200), (6_000, 6_300)]);
    }

    #[test]
    fn the_window_wraps_midnight_and_the_quiet_days_name_themselves() {
        let mut hours = vec![0u32; 24];
        hours[23] = 10;
        hours[0] = 10;
        hours[1] = 5;
        hours[12] = 1;
        assert_eq!(densest_window(&hours), (23, 3));
        assert_eq!(quiet_days(&[10, 10, 10, 10, 10, 0, 0]), "quiet on weekends");
        assert_eq!(quiet_days(&[10, 0, 10, 10, 10, 10, 10]), "quiet on Tuesdays");
        assert_eq!(quiet_days(&[10, 0, 10, 10, 10, 10, 0]), "quiet on Tuesdays and Sundays");
        assert_eq!(quiet_days(&[0, 0, 0, 0, 0, 10, 10]), "mostly on weekends");
        assert_eq!(quiet_days(&[10, 10, 0, 0, 0, 0, 10]), "mostly on Mondays, Tuesdays and Sundays");
        assert_eq!(quiet_days(&[5, 5, 5, 5, 5, 5, 5]), "every day of the week");
        assert_eq!(period(21, 3), "in the evenings");
        assert_eq!(period(23, 3), "late at night");
        assert_eq!(period(7, 4), "in the mornings");
    }
}
