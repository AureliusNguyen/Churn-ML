/**
 * Editorial section break: a centered three-asterisk asterism between
 * thin rules. Replaces a generic 1px hairline border for section seams.
 */
export function Asterism() {
  return (
    <div className="my-2 py-3" role="separator" aria-hidden>
      <div className="asterism">* * *</div>
    </div>
  );
}
