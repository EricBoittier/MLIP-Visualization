// Starting up. A copy of a page cached from before a deploy asks for bundles (and a worker) that the deploy
// removed; reloading once fetches the current build. The inline script in each page's <head> does the same
// for files that fail before any of this code runs, with the same key.
const KEY = 'mlipviz-stale';

/** Reload once to fetch the current build; false if that was already tried in this tab. */
export function reloadOnce(): boolean {
  try {
    if (sessionStorage.getItem(KEY)) return false;
    sessionStorage.setItem(KEY, '1');
  } catch { return false; }
  location.reload();
  return true;
}

/** Watch a worker for failing to load (a script the deploy removed), as opposed to failing while it runs;
 *  reload once for the first, `onFail` if that does not help. Returns a function to call once it answers. */
export function watchWorker(w: Worker, onFail: () => void) {
  let up = false;
  w.addEventListener('error', (e) => {
    if (up || (e instanceof ErrorEvent && e.message)) return; // a runtime error: the page reports it
    if (!reloadOnce()) onFail();
  });
  return () => { up = true; booted(); };
}

/** The app is up: a stale page later in this tab may reload again. */
export function booted() {
  try { sessionStorage.removeItem(KEY); } catch { /* nothing to clear */ }
}
