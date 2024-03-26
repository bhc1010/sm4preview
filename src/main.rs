use std::process::Command;
use std::path::PathBuf;

use notify::{event::EventKind, Config, RecommendedWatcher, RecursiveMode, Watcher};
use std::path::Path;

fn main() {
    let path = String::from(r"C:\Users\spmuser\OneDrive - USNH\Hollen Lab\Data\LEWIS");
    println!("Watching {}", path);
    println!("{}", "#".repeat(9 + path.len()));

    // Start watcher
    if let Err(e) = watch(path) {
        println!("error: {:?}", e)
    }
}

fn watch<P: AsRef<Path>>(path: P) -> notify::Result<()> {
    let (tx, rx) = std::sync::mpsc::channel();

    // Automatically select the best implementation for your platform.
    // You can also access each implementation directly e.g. INotifyWatcher.
    let mut watcher = RecommendedWatcher::new(tx, Config::default())?;

    // Add a path to be watched. All files and directories at that path and
    // below will be monitored for changes.
    watcher.watch(path.as_ref(), RecursiveMode::Recursive)?;

    for res in rx {
        match res {
            Ok(event) => {
                if let EventKind::Create(_) = event.kind {
                    let fpath: &PathBuf = &event.paths[0];
                    if let Some(ext) = fpath.extension() {
                        if ext == "sm4" || ext == "nc" {
                            if let Some(fname) = fpath.to_str() {
                                println!("Generating preview for {fname}");
                                
                                Command::new("python")
                                .args([r"C:\Users\spmuser\OneDrive - USNH\Hollen Lab\Resources and Programs\Python\sm4preview\src\run.py", fname])
                                .spawn()
                                .expect("Error running python instance.");
                            }
                        }
                    }
                }
            }
            Err(e) => println!("watch error: {:?}", e),
        }
    }

    Ok(())
}
