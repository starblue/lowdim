use core::cmp::Ordering;
use core::cmp::Ordering::*;

/// Utility function for chaining partial orderings.
///
/// Should probably be defined in [std::cmp::ordering].
pub fn partial_then(po0: Option<Ordering>, po1: Option<Ordering>) -> Option<Ordering> {
    match (po0, po1) {
        (None, _) => None,
        (_, None) => None,
        (Some(o0), Some(o1)) => match (o0, o1) {
            (Equal, Equal) => Some(Equal),

            (Less, Less) => Some(Less),
            (Less, Equal) => Some(Less),
            (Equal, Less) => Some(Less),

            (Greater, Greater) => Some(Greater),
            (Greater, Equal) => Some(Greater),
            (Equal, Greater) => Some(Greater),

            (Less, Greater) => None,
            (Greater, Less) => None,
        },
    }
}

/// Utility function for chaining total orderings lexicographically.
///
/// That is, the first component is most significant, and only in case
/// of equality the second component is considered.
///
/// Should probably be defined in [std::cmp::ordering].
pub fn lex_then(o0: Ordering, o1: Ordering) -> Ordering {
    match (o0, o1) {
        (Less, _) => Less,
        (Greater, _) => Greater,
        (Equal, o1) => o1,
    }
}
