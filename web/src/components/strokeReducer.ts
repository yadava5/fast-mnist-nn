/**
 * Undo/redo reducer for perfect-freehand drawing strokes.
 *
 * Shape: { past, present, future }
 * - `past` holds snapshots of `present` before each stroke finishes,
 *   so that UNDO always rolls back a full stroke (not individual
 *   pointermove samples).
 * - `present` is the array of strokes the canvas currently renders.
 *   The last entry is the in-progress stroke while the pointer is down.
 * - `future` holds undone snapshots so REDO can replay them.
 *
 * Why not push every point? Pointermove fires dozens of times per second.
 * Snapshotting the whole stroke list on each sample would balloon memory
 * and make UNDO feel broken (one stroke = hundreds of undo hops).
 */

export interface StrokePoint {
  x: number;
  y: number;
  pressure: number;
  t: number;
}

export interface Stroke {
  id: string;
  points: StrokePoint[];
  pointerType: 'mouse' | 'touch' | 'pen';
}

export interface StrokeState {
  past: Stroke[][];
  present: Stroke[];
  future: Stroke[][];
}

export type StrokeAction =
  | { type: 'BEGIN_STROKE'; stroke: Stroke }
  | { type: 'EXTEND_STROKE'; points: StrokePoint[] }
  | { type: 'END_STROKE' }
  | { type: 'UNDO' }
  | { type: 'REDO' }
  | { type: 'CLEAR' };

export const initialStrokeState: StrokeState = {
  past: [],
  present: [],
  future: [],
};

export function strokeReducer(state: StrokeState, action: StrokeAction): StrokeState {
  switch (action.type) {
    case 'BEGIN_STROKE': {
      // Snapshot `present` before we start mutating it, so UNDO can
      // roll back this whole stroke in one hop.
      return {
        past: [...state.past, state.present],
        present: [...state.present, action.stroke],
        future: [],
      };
    }

    case 'EXTEND_STROKE': {
      if (state.present.length === 0) return state;
      const last = state.present[state.present.length - 1];
      const updated: Stroke = {
        ...last,
        points: [...last.points, ...action.points],
      };
      return {
        ...state,
        present: [...state.present.slice(0, -1), updated],
      };
    }

    case 'END_STROKE': {
      // Nothing to do — BEGIN_STROKE already snapshotted for undo.
      // Kept as an explicit action so callers can hook side effects.
      return state;
    }

    case 'UNDO': {
      if (state.past.length === 0) return state;
      const previous = state.past[state.past.length - 1];
      return {
        past: state.past.slice(0, -1),
        present: previous,
        future: [state.present, ...state.future],
      };
    }

    case 'REDO': {
      if (state.future.length === 0) return state;
      const [next, ...rest] = state.future;
      return {
        past: [...state.past, state.present],
        present: next,
        future: rest,
      };
    }

    case 'CLEAR': {
      if (state.present.length === 0) return state;
      return {
        past: [...state.past, state.present],
        present: [],
        future: [],
      };
    }

    default:
      return state;
  }
}
