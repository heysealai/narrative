//! Zone-aware clock helpers: the now stamp a prompt shows, and the date the
//! harvester answers with, resolved in the user's zone. Used by harvest and rhythm.

use chrono::{NaiveDate, NaiveDateTime, TimeZone};
use chrono_tz::Tz;

pub fn zone(name: &str) -> Option<Tz> {
    name.parse().ok()
}

pub fn stamp(now: u64, timezone: Option<&str>) -> String {
    let local = timezone
        .and_then(zone)
        .and_then(|tz| tz.timestamp_opt(now as i64, 0).single())
        .map(|t| format!("{} ({})", t.format("%A %Y-%m-%d %H:%M %Z"), timezone.unwrap_or_default()));
    local.unwrap_or_else(|| {
        chrono::Utc
            .timestamp_opt(now as i64, 0)
            .single()
            .map(|t| t.format("%A %Y-%m-%d %H:%M UTC").to_string())
            .unwrap_or_else(|| format!("unix {now}"))
    })
}

pub fn parse_local(text: &str, timezone: Option<&str>) -> Option<u64> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    let (text, utc) = match text.strip_suffix('Z') {
        Some(t) => (t, true),
        None => (text, false),
    };
    let naive = ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M"]
        .iter()
        .find_map(|f| NaiveDateTime::parse_from_str(text, f).ok())
        .or_else(|| NaiveDate::parse_from_str(text, "%Y-%m-%d").ok().and_then(|d| d.and_hms_opt(12, 0, 0)))
        .or_else(|| NaiveDate::parse_from_str(&format!("{text}-15"), "%Y-%m-%d").ok().and_then(|d| d.and_hms_opt(12, 0, 0)))?;
    let secs = match (utc, timezone.and_then(zone)) {
        (false, Some(tz)) => tz.from_local_datetime(&naive).earliest()?.timestamp(),
        _ => naive.and_utc().timestamp(),
    };
    u64::try_from(secs).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_date_resolves_to_local_noon_and_a_time_to_that_local_time() {
        let noon_tokyo = parse_local("2026-09-05", Some("Asia/Tokyo")).unwrap();
        assert_eq!(noon_tokyo, 1_788_577_200, "2026-09-05 12:00 JST is 03:00 UTC");
        let noon_utc = parse_local("2026-09-05", None).unwrap();
        assert_eq!(noon_utc, 1_788_609_600);
        assert_eq!(parse_local("2026-09-05 23:22", Some("Asia/Tokyo")), parse_local("2026-09-05T14:22Z", Some("Asia/Tokyo")));
        assert_eq!(parse_local("2026-09-05T23:22:30", None), Some(noon_utc + 11 * 3_600 + 22 * 60 + 30));
        assert_eq!(parse_local("2026-09", None), parse_local("2026-09-15", None), "a month lands mid-month");
    }

    #[test]
    fn nonsense_resolves_to_nothing() {
        assert_eq!(parse_local("", Some("UTC")), None);
        assert_eq!(parse_local("last month", Some("UTC")), None);
        assert_eq!(parse_local("1788582600", Some("UTC")), None, "unix seconds are not a date");
        assert_eq!(parse_local("2026-13-40", Some("UTC")), None);
    }

    #[test]
    fn the_stamp_names_the_weekday_and_the_zone() {
        let stamp_tokyo = stamp(1_788_618_600, Some("Asia/Tokyo"));
        assert_eq!(stamp_tokyo, "Saturday 2026-09-05 23:30 JST (Asia/Tokyo)");
        assert_eq!(stamp(1_788_618_600, None), "Saturday 2026-09-05 14:30 UTC");
        assert_eq!(stamp(1_788_618_600, Some("Mars/Olympus")), "Saturday 2026-09-05 14:30 UTC");
    }
}
