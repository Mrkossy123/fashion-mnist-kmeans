import argparse
import numpy as np


# ============================================================
# K-means για Fashion-MNIST
# Καλύπτει το κομμάτι της εκφώνησης που αφορά K-means:
# - μόνο train set
# - R1: flattened εικόνα 28x28 -> 784
# - R2: histogram φωτεινότητας με M bins
# - αποστάσεις: L2, L1, cosine
# - αξιολόγηση: Purity και F-measure
# ============================================================


CLASS_NAMES = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot",
}


def load_balanced_train_set(samples_per_class=100, seed=42):
    """
    Φορτώνει ΜΟΝΟ το training set του Fashion-MNIST.
    Αν δοθεί samples_per_class, παίρνει ισοκατανεμημένο υποσύνολο ανά κατηγορία,
    όπως προτείνει η εκφώνηση για να τρέχει σε λογικό χρόνο.
    """
    try:
        from tensorflow.keras.datasets import fashion_mnist
    except Exception:
        try:
            from keras.datasets import fashion_mnist
        except Exception as exc:
            raise ImportError(
                "Δεν βρέθηκε tensorflow/keras για φόρτωση του Fashion-MNIST. \
Εγκατέστησε tensorflow ή τρέξε τον κώδικα σε περιβάλλον που υποστηρίζει keras.datasets."
            ) from exc

    (x_train, y_train), _ = fashion_mnist.load_data()

    if samples_per_class is None or samples_per_class <= 0:
        return x_train, y_train

    rng = np.random.default_rng(seed)
    selected_indices = []

    for cls in range(10):
        cls_indices = np.where(y_train == cls)[0]
        if samples_per_class > len(cls_indices):
            raise ValueError(
                f"samples_per_class={samples_per_class} > διαθέσιμα δείγματα της κλάσης {cls}"
            )
        chosen = rng.choice(cls_indices, size=samples_per_class, replace=False)
        selected_indices.extend(chosen.tolist())

    selected_indices = np.array(selected_indices)
    rng.shuffle(selected_indices)

    return x_train[selected_indices], y_train[selected_indices]


def build_representation(images, representation="R1", bins=32):
    """
    R1: flatten + normalization στο [0,1]
    R2: histogram φωτεινότητας με bins bins, κανονικοποιημένο ώστε να είναι κατανομή
    """
    images = images.astype(np.float32) / 255.0

    if representation == "R1":
        return images.reshape(images.shape[0], -1)

    if representation == "R2":
        histograms = []
        for img in images:
            hist, _ = np.histogram(img.ravel(), bins=bins, range=(0.0, 1.0), density=False)
            hist = hist.astype(np.float32)
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist /= hist_sum
            histograms.append(hist)
        return np.array(histograms, dtype=np.float32)

    raise ValueError("representation πρέπει να είναι 'R1' ή 'R2'")


def initialize_centers(X, k, seed=42):
    rng = np.random.default_rng(seed)
    if k > len(X):
        raise ValueError("Το k δεν μπορεί να είναι μεγαλύτερο από το πλήθος των δειγμάτων")
    indices = rng.choice(len(X), size=k, replace=False)
    return X[indices].copy()


def batch_distances(X_batch, centers, metric):
    """
    Επιστρέφει πίνακα αποστάσεων [batch_size, k].
    """
    if metric == "l2":
        x_sq = np.sum(X_batch * X_batch, axis=1, keepdims=True)
        c_sq = np.sum(centers * centers, axis=1)
        d_sq = x_sq + c_sq - 2.0 * (X_batch @ centers.T)
        d_sq = np.maximum(d_sq, 0.0)
        return np.sqrt(d_sq)

    if metric == "l1":
        return np.sum(np.abs(X_batch[:, None, :] - centers[None, :, :]), axis=2)

    if metric == "cosine":
        x_norm = np.linalg.norm(X_batch, axis=1, keepdims=True)
        c_norm = np.linalg.norm(centers, axis=1, keepdims=True).T
        denom = np.maximum(x_norm * c_norm, 1e-12)
        similarity = (X_batch @ centers.T) / denom
        similarity = np.clip(similarity, -1.0, 1.0)
        return 1.0 - similarity

    raise ValueError("metric πρέπει να είναι 'l2', 'l1' ή 'cosine'")


def assign_clusters(X, centers, metric="l2", batch_size=512):
    assignments = np.empty(len(X), dtype=np.int32)
    min_distances = np.empty(len(X), dtype=np.float32)

    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        dist = batch_distances(X[start:end], centers, metric)
        assignments[start:end] = np.argmin(dist, axis=1)
        min_distances[start:end] = dist[np.arange(end - start), assignments[start:end]]

    return assignments, min_distances


def recompute_centers(X, assignments, k, old_centers):
    new_centers = old_centers.copy()

    for cluster_id in range(k):
        members = X[assignments == cluster_id]
        if len(members) > 0:
            new_centers[cluster_id] = np.mean(members, axis=0)
        # Αν το cluster μείνει κενό, κρατάμε το παλιό κέντρο.

    return new_centers


def build_clusters(assignments, k):
    return [np.where(assignments == cluster_id)[0] for cluster_id in range(k)]


def majority_label(labels):
    if len(labels) == 0:
        return -1
    counts = np.bincount(labels, minlength=10)
    return int(np.argmax(counts))


def purity_score(assignments, y_true, k):
    total_correct = 0
    mapping = {}

    for cluster_id in range(k):
        cluster_labels = y_true[assignments == cluster_id]
        if len(cluster_labels) == 0:
            mapping[cluster_id] = -1
            continue

        maj = majority_label(cluster_labels)
        mapping[cluster_id] = maj
        total_correct += np.sum(cluster_labels == maj)

    purity = total_correct / len(y_true)
    return purity, mapping


def f_measure_score(assignments, y_true, k):
    """
    Για κάθε cluster παίρνουμε ως κατηγορία την πλειοψηφούσα κλάση.
    Μετά:
    TP: στοιχεία του cluster με την πλειοψηφούσα κλάση
    FP: στοιχεία του cluster με άλλη κλάση
    FN: στοιχεία εκτός cluster με την πλειοψηφούσα κλάση

    Επιστρέφουμε τον μέσο όρο των F1 των K clusters.
    """
    cluster_f1 = []
    details = {}

    for cluster_id in range(k):
        in_cluster = assignments == cluster_id
        cluster_labels = y_true[in_cluster]

        if len(cluster_labels) == 0:
            cluster_f1.append(0.0)
            details[cluster_id] = {
                "cluster_label": -1,
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "F1": 0.0,
            }
            continue

        maj = majority_label(cluster_labels)

        tp = int(np.sum(cluster_labels == maj))
        fp = int(np.sum(cluster_labels != maj))
        fn = int(np.sum((~in_cluster) & (y_true == maj)))

        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        cluster_f1.append(f1)

        details[cluster_id] = {
            "cluster_label": maj,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "F1": f1,
        }

    return float(np.mean(cluster_f1)), details


def kmeans(
    X,
    k=10,
    metric="l2",
    max_iter=50,
    tol=1e-6,
    seed=42,
    batch_size=512,
    verbose=True,
):
    """
    K-means από το μηδέν.
    Επιστρέφει assignments, centers, clusters, inertia, iterations.
    """
    centers = initialize_centers(X, k, seed=seed)
    prev_assignments = None

    for iteration in range(1, max_iter + 1):
        assignments, min_distances = assign_clusters(
            X, centers, metric=metric, batch_size=batch_size
        )
        new_centers = recompute_centers(X, assignments, k, centers)

        center_shift = float(np.linalg.norm(new_centers - centers))
        inertia = float(np.sum(min_distances ** 2))

        if verbose:
            print(
                f"[Iter {iteration:02d}] inertia={inertia:.6f} | center_shift={center_shift:.6f}"
            )

        stop_by_assignments = (
            prev_assignments is not None and np.array_equal(assignments, prev_assignments)
        )
        stop_by_shift = center_shift <= tol

        centers = new_centers
        prev_assignments = assignments.copy()

        if stop_by_assignments or stop_by_shift:
            break

    clusters = build_clusters(assignments, k)
    return {
        "assignments": assignments,
        "centers": centers,
        "clusters": clusters,
        "inertia": inertia,
        "iterations": iteration,
    }


def print_cluster_report(assignments, y_true, k):
    purity, cluster_to_label = purity_score(assignments, y_true, k)
    f_measure, f_details = f_measure_score(assignments, y_true, k)

    print("\n================= ΑΠΟΤΕΛΕΣΜΑΤΑ =================")
    print(f"Purity     : {purity:.6f}")
    print(f"F-measure  : {f_measure:.6f}")
    print("================================================")

    for cluster_id in range(k):
        idx = np.where(assignments == cluster_id)[0]
        cluster_size = len(idx)
        label_id = cluster_to_label[cluster_id]
        label_name = CLASS_NAMES.get(label_id, "EMPTY") if label_id != -1 else "EMPTY"

        print(f"\nCluster {cluster_id}")
        print(f"  Μέγεθος            : {cluster_size}")
        print(f"  Πλειοψηφούσα κλάση : {label_id} ({label_name})")

        if cluster_size > 0:
            counts = np.bincount(y_true[idx], minlength=10)
            print("  Κατανομή labels    :", counts.tolist())

        d = f_details[cluster_id]
        print(
            f"  TP={d['TP']} | FP={d['FP']} | FN={d['FN']} | F1={d['F1']:.6f}"
        )


def run_experiment(args):
    print("Φόρτωση Fashion-MNIST (train set μόνο)...")
    x_train, y_train = load_balanced_train_set(
        samples_per_class=args.samples_per_class,
        seed=args.seed,
    )

    print(f"Σχήμα x_train: {x_train.shape}")
    print(f"Σχήμα y_train: {y_train.shape}")
    print(f"Representation: {args.representation}")
    print(f"Distance      : {args.distance}")
    print(f"K             : {args.k}")

    if args.representation == "R2":
        print(f"Histogram bins: {args.bins}")

    X = build_representation(
        x_train,
        representation=args.representation,
        bins=args.bins,
    )

    print(f"Τελικό σχήμα χαρακτηριστικών: {X.shape}")

    result = kmeans(
        X,
        k=args.k,
        metric=args.distance,
        max_iter=args.max_iter,
        tol=args.tol,
        seed=args.seed,
        batch_size=args.batch_size,
        verbose=not args.quiet,
    )

    print(f"\nΣύγκλιση σε {result['iterations']} επαναλήψεις")
    print(f"Τελικό inertia: {result['inertia']:.6f}")

    print_cluster_report(result["assignments"], y_train, args.k)



def parse_args():
    parser = argparse.ArgumentParser(
        description="K-means από το μηδέν για Fashion-MNIST"
    )
    parser.add_argument("--representation", choices=["R1", "R2"], default="R1")
    parser.add_argument("--distance", choices=["l2", "l1", "cosine"], default="l2")
    parser.add_argument("--bins", type=int, default=32, help="Χρήσιμο μόνο όταν representation=R2")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=100,
        help="Ισοκατανεμημένο υποσύνολο train ανά κλάση. Βάλε 0 ή αρνητικό για όλο το train set.",
    )
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
