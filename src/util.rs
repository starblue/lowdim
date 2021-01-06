use core::cmp::Ordering;
use core::cmp::Ordering::*;

/// Utility function for chaining partial orderings.
///
/// Should be in std::cmp::ordering .
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
