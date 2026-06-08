import type { MutableRefObject, Ref } from "react";

/** Combine multiple React refs onto one DOM node (lab-style direct video ref). */
export function mergeRefs<T>(...refs: Array<Ref<T> | undefined>): (value: T | null) => void {
  return (value) => {
    for (const ref of refs) {
      if (!ref) continue;
      if (typeof ref === "function") ref(value);
      else (ref as MutableRefObject<T | null>).current = value;
    }
  };
}
