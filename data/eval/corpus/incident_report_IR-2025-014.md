# Incident Report IR-2025-014

**Severity:** SEV-2
**Status:** Resolved
**Detected:** 2025-08-14 03:22 UTC
**Resolved:** 2025-08-21 19:45 UTC
**Author:** Daniel Okafor (CTO), with input from the SR-2 firmware team

## Summary

Three of seven SR-2 units deployed at FreightCo Distribution Center #7 (Memphis) experienced gait instability beginning approximately four hours after firmware update **v3.4.1** was rolled out on the night shift of 2025-08-13. Affected units exhibited intermittent ankle dorsiflexion oscillation under load, leading to two documented falls (no injuries; one shelving unit damaged). The remaining four units at DC#7, and all units at DC#3 and DC#5 (which had not yet received the v3.4.1 update), were unaffected.

## Timeline

- **2025-08-13 22:00 UTC** — v3.4.1 staged rollout begins at DC#7 (7 units).
- **2025-08-14 02:55 UTC** — first unit (serial SR2-0044) flagged abnormal IMU variance.
- **2025-08-14 03:22 UTC** — automated alert raised; on-call engineer paged.
- **2025-08-14 03:51 UTC** — second fall event recorded for SR2-0049.
- **2025-08-14 04:10 UTC** — fleet remotely commanded to standby; v3.3.8 rollback initiated.
- **2025-08-14 06:30 UTC** — all DC#7 units back on v3.3.8 and operational.
- **2025-08-21 19:45 UTC** — hotfix v3.4.2 deployed across the fleet after RCA and regression testing.

## Root cause

Under specific payload-and-stride combinations (payload >18 kg with stride length >62 cm), the ankle torque PID controller produced a 32-bit floating-point intermediate term that, when accumulated, exceeded the representable range of the downstream fixed-point conversion, wrapping to a large negative value. The downstream actuator interpreted this as a maximum-magnitude reverse torque command for one control cycle (~2 ms), producing the observed ankle oscillation.

The bug was introduced in the v3.4.0 → v3.4.1 ankle-controller refactor. Existing unit tests covered the controller in isolation but did not test the conversion bridge under extreme inputs.

## Customer impact

- **14 calendar days of degraded service** at DC#7 (3 of 7 units offline pending hotfix).
- **No human injury.** One shelving unit at DC#7 sustained cosmetic damage.
- **$420,000 in service credits** issued to FreightCo under the SLA terms of the MSA. Credits were recognized in Q3 2025.

## Action items

1. Add fuzz testing across the controller-to-actuator conversion bridge. **Owner:** firmware lead. **Due:** end of Q4 2025.
2. Delay the v3.5 firmware feature release by one quarter to absorb the regression-testing expansion. **Owner:** engineering leadership.
3. Add per-cycle saturation guards on all actuator command paths. **Owner:** controls team. **Due:** v3.4.3.
4. Joint review with FreightCo of the staged-rollout protocol. **Owner:** customer success. **Status:** completed 2025-09-08.
