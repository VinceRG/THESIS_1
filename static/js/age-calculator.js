function parseLocalDate(value) {
  const parts = (value || '').split('-').map(Number);
  if (parts.length !== 3 || parts.some(Number.isNaN)) return null;
  return { year: parts[0], month: parts[1], day: parts[2] };
}

function calculateAgeFromBirthdate(birthdateValue, referenceValue) {
  const birthdate = parseLocalDate(birthdateValue);
  const reference = parseLocalDate(referenceValue);
  if (!birthdate || !reference) return null;
  let age = reference.year - birthdate.year;
  if (reference.month < birthdate.month || (reference.month === birthdate.month && reference.day < birthdate.day)) {
    age -= 1;
  }
  return age >= 0 && age <= 130 ? age : null;
}
