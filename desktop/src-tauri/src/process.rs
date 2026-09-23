//! Arbre de processus du coeur : uv -> python -> llama-server (x2).
//!
//! * Windows : un Job Object "kill on close" contient tout l'arbre. S'il faut arreter, ou si la coquille
//!   meurt (meme brutalement), Windows tue le coeur et les llama-server : la VRAM est rendue.
//! * Unix : le coeur a son propre groupe de processus. Arret = SIGTERM (uvicorn s'arrete proprement et
//!   stoppe les llama-server), puis SIGKILL sur le groupe si rien ne bouge. `--parent-pid` couvre le
//!   cas ou la coquille meurt sans avoir pu faire le menage.

use std::process::{Child, Command};

/// Prepare la commande (a appeler avant `spawn`).
pub fn configure(cmd: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        cmd.process_group(0);
    }
}

/// Poignee sur l'arbre de processus d'un coeur lance.
pub struct Tree {
    #[cfg(windows)]
    job: Option<win::Job>,
    #[cfg(unix)]
    pgid: i32,
}

impl Tree {
    /// A appeler juste apres `spawn`.
    pub fn attach(child: &Child) -> Self {
        #[cfg(windows)]
        {
            use std::os::windows::io::AsRawHandle;
            Tree { job: win::Job::containing(child.as_raw_handle()) }
        }
        #[cfg(unix)]
        {
            Tree { pgid: child.id() as i32 }
        }
    }

    /// Demande d'arret. Unix : SIGTERM au processus principal (uv relaie a python).
    /// Windows : pas de signal propre pour un processus sans console, on termine l'arbre.
    pub fn terminate(&self, child: &mut Child) {
        #[cfg(unix)]
        {
            // SAFETY: kill(2) n'a pas d'effet sur la memoire ; le processus n'est pas encore recolte
            // (wait), son pid ne peut donc pas avoir ete reattribue.
            unsafe {
                libc::kill(child.id() as i32, libc::SIGTERM);
            }
        }
        #[cfg(windows)]
        self.kill(child);
    }

    /// Arret force de tout l'arbre.
    pub fn kill(&self, child: &mut Child) {
        #[cfg(unix)]
        {
            // SAFETY: voir `terminate` ; un pid de groupe negatif vise tout le groupe du coeur.
            unsafe {
                libc::kill(-self.pgid, libc::SIGKILL);
            }
        }
        #[cfg(windows)]
        if let Some(job) = &self.job {
            job.terminate();
        }
        let _ = child.kill();
    }
}

/// Unix : SIGINT / SIGTERM (Ctrl+C dans un terminal, fermeture de session) -> `on_signal` est appele
/// hors du gestionnaire de signal, pour une sortie propre qui arrete le coeur.
#[cfg(unix)]
pub fn on_termination_signal(on_signal: impl FnOnce() + Send + 'static) {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    static RECEIVED: AtomicBool = AtomicBool::new(false);
    extern "C" fn handler(_: libc::c_int) {
        RECEIVED.store(true, Ordering::SeqCst); // seule operation faite dans le gestionnaire (async-signal-safe)
    }
    let handler: extern "C" fn(libc::c_int) = handler;
    // SAFETY: le gestionnaire ne fait qu'une ecriture atomique.
    unsafe {
        libc::signal(libc::SIGINT, handler as libc::sighandler_t);
        libc::signal(libc::SIGTERM, handler as libc::sighandler_t);
    }
    std::thread::spawn(move || {
        while !RECEIVED.load(Ordering::SeqCst) {
            std::thread::sleep(Duration::from_millis(200));
        }
        on_signal();
    });
}

#[cfg(windows)]
mod win {
    use std::ffi::c_void;
    use std::mem::{size_of, zeroed};
    use std::ptr::null;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject,
        TerminateJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    pub struct Job(HANDLE);

    // SAFETY: une poignee de Job Object peut etre utilisee depuis n'importe quel fil.
    unsafe impl Send for Job {}
    unsafe impl Sync for Job {}

    impl Job {
        /// Cree un Job "kill on close" et y place le processus. `None` si Windows refuse (on retombe
        /// alors sur `--parent-pid` et `Child::kill`).
        pub fn containing(process: *mut c_void) -> Option<Job> {
            // SAFETY: appels Win32 documentes ; la structure est un POD initialise a zero puis rempli.
            unsafe {
                let handle = CreateJobObjectW(null(), null());
                if handle.is_null() {
                    return None;
                }
                let job = Job(handle);
                let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = zeroed();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                let ok = SetInformationJobObject(
                    job.0,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const c_void,
                    size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                ) != 0
                    && AssignProcessToJobObject(job.0, process as HANDLE) != 0;
                ok.then_some(job)
            }
        }

        pub fn terminate(&self) {
            // SAFETY: poignee valide tant que `self` existe.
            unsafe {
                TerminateJobObject(self.0, 1);
            }
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            // SAFETY: fermer la derniere poignee tue les processus restants (KILL_ON_JOB_CLOSE).
            unsafe {
                CloseHandle(self.0);
            }
        }
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    fn wait_exit(child: &mut Child, timeout: Duration) -> bool {
        let deadline = Instant::now() + timeout;
        while Instant::now() < deadline {
            if let Ok(Some(_)) = child.try_wait() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        false
    }

    #[test]
    fn terminate_asks_politely() {
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "trap 'exit 0' TERM; while :; do sleep 0.05; done"]);
        configure(&mut cmd);
        let mut child = cmd.spawn().unwrap();
        let tree = Tree::attach(&child);
        std::thread::sleep(Duration::from_millis(200)); // laisse le temps d'installer le trap
        tree.terminate(&mut child);
        assert!(wait_exit(&mut child, Duration::from_secs(5)));
        assert_eq!(child.wait().unwrap().code(), Some(0)); // sortie propre, pas un SIGKILL
    }

    #[test]
    fn kill_takes_down_the_whole_group() {
        // le parent ignore SIGTERM et lance un petit-enfant dans le meme groupe
        let mut cmd = Command::new("sh");
        cmd.args(["-c", "trap '' TERM; sleep 30 & echo $!; wait"]).stdout(std::process::Stdio::piped());
        configure(&mut cmd);
        let mut child = cmd.spawn().unwrap();
        let tree = Tree::attach(&child);
        let mut line = String::new();
        std::io::BufRead::read_line(&mut std::io::BufReader::new(child.stdout.take().unwrap()), &mut line).unwrap();
        let grandchild: i32 = line.trim().parse().unwrap();
        tree.kill(&mut child);
        assert!(wait_exit(&mut child, Duration::from_secs(5)));
        let deadline = Instant::now() + Duration::from_secs(5);
        // SAFETY: kill(pid, 0) ne fait que tester l'existence du processus
        while unsafe { libc::kill(grandchild, 0) } == 0 && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(20));
        }
        assert_ne!(unsafe { libc::kill(grandchild, 0) }, 0, "le petit-enfant a survecu");
    }
}
