# -*- coding: utf-8 -*-
# This code is modified from https://github.com/bmcfee/lsd_viz
# Original code license: https://github.com/bmcfee/lsd_viz/blob/main/LICENSE.md (enclosed in external/lsd_viz)
# Reference: McFee, Brian, and Dan Ellis. "Analyzing Song Structure with Spectral Clustering." ISMIR. 2014.
import librosa
import madmom.features.beats
import numpy as np
import scipy
import sklearn.cluster
import tempfile
import os
import sox
import time
from collections import defaultdict
from .deepsim import run_deepsim_inference
from .fewshot import run_fewshot_inference


# CONSTANTS
N_LEVELS = 12
BPO = 12 * 3
N_OCTAVES = 7
DISTANCE = "cosine"
NORMALIZE_MATRICES = True
MAGICNORM = True
MAXNORM = False
LOG = False


def print_verbose(text, verbose):
    """Print text if verbose

    Parameters
    ----------
    text : string
        string to print
    verbose : bool
        If True print text
    """
    if verbose:
        print(text)


def segments_to_levels(segs_list):
    """
    Given a list of segmentation in 2D numpy array format, convert it back into
    the format returned by reindex().

    Parameters
    ----------
    segs_list: list
        List of segmentations in 2D numpy array format

    Returns
    -------
    levels: list
        List of segmentations in the format returned by reindex()
    """
    levels = []
    for segs in segs_list:
        intervals = []
        ids = []
        for row in segs:
            intervals.append(tuple(row[:2]))
            ids.append(str(int(row[2])))
        levels.append(tuple([intervals, ids]))
    return levels


def clean_segments(levels, min_duration=8, fix_level=3, verbose=False):
    """
    Given segmentation levels (in the format returned by reindex(), take segmentation level fix_level (note that levels
    start at 1, not 0) and merge out all segments shorter than min_duration. Returns the cleaned up segments as a 2D
    numpy array.

    Cleaning of short segments follows this pseudo algorithm:

    PSEUDOALG:
    while shortest_seg < min_duaration and n_segs > 1:
        if current_seg = start of song
            merge with next_seg
        elif current_seg = end of song
            merge with previous_seg
        elif id_prev_seg == id_next_seg
            merge prev + current + next
        else:
            while not solved and level is not lowest:
                look down 1 level, get id of segment with most overlap
                if lower id is same as prev or next ID, use it, we're done.
                else, repeat
                ADDITION: if we go all the way down to level 1 and still dont
                find a good ID, check which of the short segment's boundaries
                overlaps the most with boundaries at lower levels and keep that
                one (and merge short seg with the seg adjacent to the losing
                boundary).

    Parameters
    ----------
    levels: list
        List of segmentation in the format returned by reindex()
    min_duration: float
        Minimum duration to keep a segment
    fix_level:
        Segmentation level to be fixed (note: starts at 1)
    verbose:
        Verbose printouts if true.

    Returns
    -------
    segs: np.ndarray
        Cleaned segments from level fix_level as 2D numpy array
    """

    def get_segs(levs, level=3):
        """
        Utility function to get segments for a specific segmentation level in
        numpy array format. Specify the level start at 1.

        Parameters
        ----------
        levs: segmentation levels as returned by reindex()
        level: level of segmentation to return (minimum is 1).

        Returns
        -------

        """
        ts, ids = levs[level - 1][0], levs[level - 1][1]
        array = []
        for ((start, end), i) in zip(ts, ids):
            array.append([start, end, int(i)])

        return np.asarray(array)

    def durations(segs):
        """
        Given segments in the format returned by get_segs(), return an array
        with the durations of the segments

        Parameters
        ----------
        segs: np.ndarray
            Segments array in the format returned by get_segs()

        Returns
        -------
        durations: np.ndarray
            Array of segment durations

        """
        return segs[:, 1] - segs[:, 0]

    def merge_segs(segs, first_idx, last_idx, new_id):
        """
        Given segments in the format returned by get_segs(), i.e. a 2D numpy
        array, the indices if the first and last segments to merge (indexing
        starts at 0), and the id for the new merged segment, merge the segments
        and return a new segments array.


        Parameters
        ----------
        segs: np.ndarray
            Segments in the format returned by get_segs()
        first_idx: int
            Index of first segment to merge
        last_idx: int
            Index of last segment to merge
        new_id: int
            id for the merged segment

        Returns
        -------
        segs: np.ndarray
            Merged segments in 2D numpy array format
        """
        assert first_idx < last_idx
        new_segs = []

        new_start = None
        new_end = None

        for n, seg in enumerate(segs):
            if n < first_idx:
                new_segs.append(seg)
            elif n == first_idx:
                new_start = seg[0]
            elif n < last_idx:
                pass
            elif n == last_idx:
                new_end = seg[1]
                new_segs.append(np.asarray([new_start, new_end, new_id]))
            else:
                new_segs.append(seg)

        return np.asarray(new_segs)

    def get_overlap_time(s1, s2):
        """
        Get the overlap duration (in seconds) between segments s1 and s2. If the
        segments don't overlap at all a duration of 0 is returned.

        Params:
        s1/s2 = list or tuple of the form (start_time, end_time)

        Parameters
        ----------
        s1: list/np.ndarray/tuple
        s2: list/np.ndarray/tuple

        Returns
        -------
        overlap: float
            Overlap duration between s1 and s2 in seconds.
        """

        max_start = max(s1[0], s2[0])
        min_end = min(s1[1], s2[1])
        return max(min_end - max_start, 0)

    def get_down_id(minidx, segs, dsegs):
        """
        Given the segments segs at a certain segmentation level, segments at
        a lower segmentation level dsegs, and the index minidx of the shortest
        segment in segs, return the segment ID of the segment in dsegs that
        overlaps the most with the segment given by segs[minidx].

        Parameters
        ----------
        minidx: int
            Index of the segment in segs (starts at 0)
        segs: np.ndarray
            Segments in 2D numpy array format
        dsegs: np.ndarray
            Segments at a lower level than segs, in 2D numpy array format

        Returns
        -------
        downid: int
            ID of the segment in dsegs that overlaps the most (in time) with
            the segment given by segs[minidx].
        """
        seg_times = segs[minidx, :2]
        down_times = dsegs[:, :2]
        overlaps = []
        for dt in down_times:
            overlaps.append(get_overlap_time(seg_times, dt))

        max_overlap_idx = np.argmax(overlaps)
        downid = dsegs[max_overlap_idx, 2]

        return downid

    def get_boundary_overlap(levels, fix_level, minidx, max_distance=1, verbose=False):
        """
        Given a segment specified by its fix_level and minidx, count how many
        segments at lower levels have a start time that overlaps with the
        segment's start and end times.

        Parameters
        ----------
        levels: list
            Segmeentations as returned by reindex()
        fix_level: int
            Level of segmentation to consider (starts at 1)
        minidx: int
            Index of the short segment whose boundaries will be examined
        max_distance: float
            The max distance (in seconds) between two boundaries to consider
            them as overlapping.

        Returns
        -------
        boundary_overlap_start: int
            Count of boundaries at lower levels that overlap with the segment's
            start time
        boundary_overlap_end: int
            Count of boundaries at lower levels that overlap with the segment's
            end time.
        """
        segs = get_segs(levels, level=fix_level)
        # print_verbose("---------> Looking for boundary overlap for minidx: {}, start: {:.1f}, end: {:.1f}".format(
        #     minidx, segs[minidx, 0], segs[minidx, 1]), verbose)

        boundary_overlap_start = 0
        boundary_overlap_end = 0

        downlevel = fix_level - 1
        while downlevel > 0:
            dsegs = get_segs(levels, downlevel)
            for ds in dsegs:
                if np.abs(ds[0] - segs[minidx, 0]) <= max_distance:
                    # print_verbose("---------> start boundary ({:.1f}) +1 at downlevel {} at time {:.1f}".format(
                    #     segs[minidx, 0], downlevel, ds[0]), verbose)
                    boundary_overlap_start += 1
                if np.abs(ds[0] - segs[minidx, 1]) <= max_distance:
                    # print_verbose("---------> end boundary ({:.1f}) +1 at downlevel {} at time {:.1f}".format(
                    #     segs[minidx, 1], downlevel, ds[0]), verbose)
                    boundary_overlap_end += 1
            downlevel -= 1

        return boundary_overlap_start, boundary_overlap_end

    # ********** BEGINNING OF CLEAN_SEGMENTS **********
    if fix_level <= 1:
        return get_segs(levels, level=1)

    segs = get_segs(levels, level=fix_level)
    downid = None

    id_to_fix = np.max(segs[:, 2])

    # OLD STRATEGY: always fix first the shortest segment. Could lead to sob-optimal results in rare cases.
    # while min(durations(segs)) < min_duration and len(segs) > 1:  # repeat until no short segs left or just 1 seg left

    # NEW STRATEGY: first fix the small segments of the highest seg ID, then move to previous seg ID, etc., till
    # We reach seg ID 0 which is the last seg ID to fix. This gives short segments of lower IDs priority over short
    # segs of higher IDs.
    while id_to_fix >= 0 and len(segs) > 1:

        # NEW SAMPLING STRATEGY:
        if not np.any(segs[:, 2] == id_to_fix):
            id_to_fix -= 1
            continue

        rows_with_id = segs[segs[:, 2] == id_to_fix]
        id_durations = durations(rows_with_id)
        minrow = rows_with_id[np.argmin(id_durations)]
        minidx = None
        for i in range(len(segs)):
            if np.allclose(segs[i], minrow):
                minidx = i
                break

        if durations(segs)[minidx] > min_duration:
            id_to_fix -= 1
            continue

        # OLD SAMPLING STRATEGY:
        # minidx = np.argmin(durations(segs))  # find shortest seg

        downlevel = fix_level - 1  # must be inside loop so it resets at each iteration!

        if minidx == 0:  # if first seg, merge with next seg
            print_verbose("merging first (level: {}, minidx: {})".format(fix_level, minidx), verbose)
            new_id = segs[minidx + 1, 2]
            segs = merge_segs(segs, 0, 1, new_id)

        elif minidx == len(segs) - 1:  # if last seg, merge with prev seg
            print_verbose("merging last (level: {}, minidx: {})".format(fix_level, minidx), verbose)
            new_id = segs[minidx - 1, 2]
            segs = merge_segs(segs, minidx - 1, minidx, new_id)

        elif segs[minidx - 1, 2] == segs[minidx + 1, 2]:  # if seg ID same as prev and next, merge the 3
            print_verbose("merging same before/after (level: {}, minidx: {})".format(fix_level, minidx), verbose)
            new_id = segs[minidx - 1, 2]
            segs = merge_segs(segs, minidx - 1, minidx + 1, new_id)
        else:  # otherwise consult lower level for seg ID
            # GO DOWN A LEVEL FOR CONSULT
            print_verbose("consluting lower levels (this level: {}, lower level: {}, minidx: {}, id: {})".format(
                fix_level, downlevel, minidx, segs[minidx, 2]), verbose)
            solved = False
            while not solved:
                dsegs = get_segs(levels, downlevel)
                # find majority seg ID overlapping with current seg
                downid = get_down_id(minidx, segs, dsegs)
                # get neighboring IDs
                neighborids = [segs[minidx - 1, 2], segs[minidx + 1, 2]]

                print_verbose("---> this level: {}, lower level: {}, id: {}, downid: {})".format(
                    fix_level, downlevel, segs[minidx, 2], downid), verbose)

                if downid not in neighborids and downlevel > 1:  # if lower level is same ID and not 0, go down
                    downlevel -= 1
                else:  # otherwise use seg ID from lower lev and merge
                    # USE downid
                    if segs[minidx - 1, 2] == downid == segs[minidx + 1, 2]:
                        print_verbose("------> found good downid: {}, merging (prev/next)".format(downid), verbose)
                        segs = merge_segs(segs, minidx - 1, minidx + 1, downid)
                    elif downid == segs[minidx - 1, 2]:
                        print_verbose("------> found good downid: {}, merging (prev)".format(downid), verbose)
                        segs = merge_segs(segs, minidx - 1, minidx, downid)
                    elif downid == segs[minidx + 1, 2]:
                        print_verbose("------> found good downid: {}, merging (next)".format(downid), verbose)
                        segs = merge_segs(segs, minidx, minidx + 1, downid)
                    else:
                        print_verbose("------> NO good downid, looking at boundary overlap", verbose)
                        # In this case we could NOT find a good downid and have gone all the way down to level 1
                        # So, instead, we look at the short segment's start/end times, and see which one overlaps the
                        # most with start/end times at lower levels. Whichever overlaps the most is kept as a boundary
                        # and the other is merged with its adjacent section and take on its ID.
                        print_verbose("minidx: {}".format(minidx), verbose)
                        print_verbose("segs:\n{}".format(segs), verbose)
                        boundary_overlap_start, boundary_overlap_end = get_boundary_overlap(
                            levels, fix_level, minidx, max_distance=1, verbose=verbose)
                        if boundary_overlap_start > boundary_overlap_end:
                            print_verbose("---------> Start wins ({}/{}), merging next".format(
                                boundary_overlap_start, boundary_overlap_end), verbose)
                            segs = merge_segs(segs, minidx, minidx + 1, segs[minidx + 1, 2])
                        else:
                            print_verbose("---------> End wins ({}/{}), merging prev".format(
                                boundary_overlap_start, boundary_overlap_end), verbose)
                            segs = merge_segs(segs, minidx - 1, minidx, segs[minidx - 1, 2])
                        print_verbose("new segs:\n{}".format(segs), verbose)
                    solved = True

    # print_verbose("Done cleaning segments.", verbose)
    return segs


def _reindex_labels(ref_int, ref_lab, est_int, est_lab):
    """Reindex estimate labels so they maximally overlap 
    with the reference labels based on their intervals.
    
    for each estimated label:
            find the reference label that it maximally overlaps with

    Parameters
    ----------
    ref_int : list
        reference intervals
    ref_lab : list
        refertence labels
    est_int : list
        estimate intervals
    est_lab : list
        estimate labels

    Returns
    -------
    list
        Reindexed labels
    """
    score_map = defaultdict(lambda: 0)

    for r_int, r_lab in zip(ref_int, ref_lab):
        for e_int, e_lab in zip(est_int, est_lab):
            score_map[(e_lab, r_lab)] += max(0, min(e_int[1], r_int[1]) -
                                             max(e_int[0], r_int[0]))

    r_taken = set()
    e_map = dict()

    hits = [(score_map[k], k) for k in score_map]
    hits = sorted(hits, reverse=True)

    while hits:
        cand_v, (e_lab, r_lab) = hits.pop(0)
        if r_lab in r_taken or e_lab in e_map:
            continue
        e_map[e_lab] = r_lab
        r_taken.add(r_lab)

    # Anything left over is unused
    unused = set(est_lab) - set(ref_lab)

    for e, u in zip(set(est_lab) - set(e_map.keys()), unused):
        e_map[e] = u

    return [e_map[e] for e in est_lab]


def reindex(hierarchy):
    """At each segmentation level, match the section labels (IDs)
    to those with which they most overlap at the level below.
    This gives consistent section labels across segmentation levels,
    which is both helpful to the user and necessary for the multi-level
    section fusion algorithm.

    Parameters
    ----------
    hierarchy : list
        Multi-level segmentation

    Returns
    -------
    list
        Reindexed multi-level segmentation
    """
    new_hier = [hierarchy[0]]
    for i in range(1, len(hierarchy)):
        ints, labs = hierarchy[i]
        labs = _reindex_labels(new_hier[i-1][0], new_hier[i-1][1], ints, labs)
        new_hier.append((ints, labs))

    return new_hier


def normalize_matrix(X, maxnorm=False):
    """Normalize matrix by dividing it by its norm

    Parameters
    ----------
    X : np.ndarray
        matrix
    maxnorm : bool, optional
        If True normalize matrix by its max instead of norm (by default False)

    Returns
    -------
    [type]
        [description]
    """
    if maxnorm:
        X /= X.max() + np.finfo(np.float64).eps
    else:
        X /= np.linalg.norm(X) + np.finfo(np.float64).eps
    return X


def embed_beats(A_rep, 
                A_loc, 
                Hsync, 
                mu=0.5,
                gamma=0.5, 
                recsmooth=9, 
                recwidth=9, 
                evecsmooth=9, 
                normalize_matrices=True,
                distance="cosine", 
                maxnorm=False):
    """[summary]

    Parameters
    ----------
    A_rep : np.ndarray
        deepsim recurrence matrix
    A_loc : np.ndarray
        fewshot (or mfcc) recurrence matrix
    Hsync : np.ndarray
        CQT (harmony) recurrency matrix
    mu : float, optional
        [description], by default 0.5
    gamma : float, optional
        [description], by default 0.5
    recsmooth : int, optional
        [description], by default 9
    recwidth : int, optional
        [description], by default 9
    evecsmooth : int, optional
        [description], by default 9
    normalize_matrices : bool, optional
        [description], by default True
    distance : str, optional
        [description], by default "cosine"
    maxnorm : bool, optional
        [description], by default False

    Returns
    -------
    np.ndarray
        Feature embedding
    """
    # Build recurrence graph from deepsim
    R = librosa.segment.recurrence_matrix(
        A_rep, 
        width=min(recwidth, min(A_rep.shape)),
        mode='affinity',
        metric=distance,
        sym=True)

    # Enhance diagonals with a median filter (LSD paper Equation 2)
    df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    Rf = df(R, size=(1, recsmooth))

    # Build local graph from fewshot (or MFCC)
    path_distance = np.sum(np.diff(A_loc, axis=1)**2, axis=0)
    sigma = np.median(path_distance)
    path_sim = np.exp(-path_distance / sigma)
    R_path = np.diag(path_sim, k=1) + np.diag(path_sim, k=-1)

    ##########################################################
    # And compute the balanced combination (LSD paper Equations 6, 7, 9)
    # deg_path = np.sum(R_path, axis=1)
    # deg_rec = np.sum(Rf, axis=1)
    # if mu is None:
    #     mu = deg_path.dot(deg_path + deg_rec) / np.sum((deg_path + deg_rec)**2)  # orig
    # print_verbose("mu: {}".format(mu), LOG)
    ##########################################################

    # Build recurrence graph from chroma
    chroma_R = librosa.segment.recurrence_matrix(
        Hsync, 
        width=min(recwidth, min(Hsync.shape)),
        mode='affinity',
        metric='euclidean',
        sym=True)

    # Enhance diagonals with a median filter (LSD paper Equation 2)
    chroma_df = librosa.segment.timelag_filter(scipy.ndimage.median_filter)
    chroma_Rf = chroma_df(chroma_R, size=(1, recsmooth))

    m = np.min([Rf.shape[0], R_path.shape[0], chroma_Rf.shape[0]])
    Rf = Rf[:m, :m]
    R_path = R_path[:m, :m]
    chroma_Rf = chroma_Rf[:m, :m]
    
    if normalize_matrices:
        Rf = normalize_matrix(Rf, maxnorm=maxnorm)
        chroma_Rf = normalize_matrix(chroma_Rf, maxnorm=maxnorm)
        R_path = normalize_matrix(R_path, maxnorm=maxnorm)
        
        A = (mu * gamma) * Rf + (mu * (1-gamma)) * chroma_Rf + (1 - mu) * R_path  # norm
    else:
        A = mu * Rf + mu * chroma_Rf + (1 - 2 * mu) * R_path  # no norm


    #####################################################
    # Now let's compute the normalized Laplacian (LSD paper Eq. 10)
    L = scipy.sparse.csgraph.laplacian(A, normed=True)

    # and its spectral decomposition
    evals, evecs = scipy.linalg.eigh(L)

    # We can clean this up further with a median filter.
    # This can help smooth over small discontinuities
    evecs = scipy.ndimage.median_filter(evecs, size=(evecsmooth, 1))

    return evecs


def make_single_section(audio_duration, n_levels):
    """Create one section at all segmentation levels that spans the entire audio duration.

    Parameters
    ----------
    audio_duration : float
        Duration of audio file in seconds
    n_levels : int
        Number of segmentation levels to return

    Returns
    -------
    segs: list
        Segmentation levels
    """
    segs = []
    seg_start = 0
    seg_end = audio_duration
    for _ in range(1, n_levels + 1):
        segs.append(([seg_start, seg_end], ["0"]))
    return segs


def cluster(evecs, Cnorm, k, beat_times):
    """cluster embedding to produce sections

    Parameters
    ----------
    evecs : np.ndarray
        embedding to cluster
    Cnorm : [type]
        [description]
    k : int
        number of clusters
    beat_times : np.ndarray
        array of beat times in frames

    Returns
    -------
    [type]
        [description]
    """

    # cluster
    X = evecs[:, :k] / (Cnorm[:, k - 1:k] + np.finfo(np.float64).eps)
    KM = sklearn.cluster.KMeans(n_clusters=k, n_init=50, max_iter=500)
    seg_ids = KM.fit_predict(X)

    # Locate segment boundaries from the label sequence
    bound_beats = 1 + np.flatnonzero(seg_ids[:-1] != seg_ids[1:])

    # Count beats 0 as a boundary
    bound_beats = librosa.util.fix_frames(bound_beats, x_min=0)

    # Compute the segment label for each boundary
    bound_segs = list(seg_ids[bound_beats])

    # Convert beat indices to times in seconds
    bound_times = beat_times[bound_beats]

    # Tack on the end-time
    bound_times = list(np.append(bound_times, beat_times[-1]))

    ivals, labs = [], []
    for interval, label in zip(zip(bound_times, bound_times[1:]), bound_segs):
        ivals.append(interval)
        labs.append(str(label))

    return ivals, labs


def segment_features(features,
                     min_duration=8,
                     mu=0.5,
                     gamma=0.5,
                     recurrence_width=9,
                     recurrence_smooth=9,                                           
                     eigenvec_smooth=9):
    
    # unpack features
    Csync = features['Csync']
    Msync = features['Msync']
    Hsync = features['Hsync']
    beat_times = features['beat_times']
    audio_duration = features['audio_duration']
    
    # safety check
    if Csync is None or Msync is None or Hsync is None:
        segs = make_single_section(audio_duration, N_LEVELS)
        return segs

    # Embed beat-synchronized feature matrices via LSD
    embedding = embed_beats(Csync, 
                            Msync, 
                            Hsync, 
                            mu=mu, 
                            gamma=gamma, 
                            recsmooth=recurrence_smooth,
                            recwidth=recurrence_width, 
                            evecsmooth=eigenvec_smooth, 
                            normalize_matrices=NORMALIZE_MATRICES,
                            distance=DISTANCE, 
                            maxnorm=MAXNORM)

    # Cluster to obtain segmentations at k=1..N_LEVELS levels
    Cnorm = np.cumsum(embedding**2, axis=1)**0.5    
    segmentations = []
    for k in range(1, min(N_LEVELS+1, embedding.shape[0]+1)):
        segmentations.append(cluster(embedding, Cnorm, k, beat_times))

    # Reindex section IDs for multi-level consistency
    levels = reindex(segmentations)
    
    # If min_duration is set, apply multi-level SECTION FUSION 
    # to remove short sections
    fixed_levels = None
    if min_duration is None:
        fixed_levels = levels
    else:
        segs_list = []
        for i in range(1, len(levels) + 1):
            segs_list.append(clean_segments(levels, 
                                            min_duration=min_duration, 
                                            fix_level=i, 
                                            verbose=False))
        
        fixed_levels = segments_to_levels(segs_list)
    
    return fixed_levels


def madmom_beats(audiofile):
    """Compute madmom beat times. Approach described in:
    
    F. Korzeniowski, S. Böck and G. Widmer, “Probabilistic Extraction of Beat Positions from a 
    Beat Activation Function”, In ISMIR 2014.

    Parameters
    ----------
    audiofile : str
        Path to audiofile for estimating beat times

    Returns
    -------
    list
        List of beat times in seconds
    """
    proc = madmom.features.beats.CRFBeatDetectionProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(audiofile)
    beat_times = np.asarray(proc(act))
    return beat_times


def load_beats(beats_file, audio_duration):
    """Load (e.g. madmom) beats file and return beats as frames (beats) and 
    in seconds (beat_times), where the latter has times 0, audio_duration
    added to it as position 0, -1 respectively.

    Beats file must store one beat timestamp per row, in seconds.

    Parameters
    ----------
    beats_file : str
        path to txt file storing beats as one beat timestamp per row, in seconds
    audio_duration : float
        duration of audio file in seconds

    Returns
    -------
    beats : np.ndarray
        beat times in frames
    beat_times : np.ndarray
        beat times in seconds
    """
    beat_times = np.loadtxt(beats_file)

    beats = librosa.time_to_frames(beat_times, sr=22050, hop_length=512)
    if beat_times[0] > 0:
        beat_times = np.insert(beat_times, 0, 0)
    if beat_times[-1] < audio_duration:
        beat_times = np.append(beat_times, audio_duration)

    return beats, beat_times


def make_beat_sync_features(filename, 
                            deepsim_model, 
                            fewshot_model,
                            beats_alg="madmom",
                            beats_file=None,
                            use_mfcc=False,
                            magicnorm=True):
    """Compute embeddings/features from audio file and beat-sync them

    Parameters
    ----------
    filename : str
        path to audio file
    deepsim_model : [type]
        Deepsim model object
    fewshot_model : [type]
        Fewshot model object
    beats_alg : str, optional
        Beat tracking algorithm to use, "madmom" or "librosa", by default "madmom"
    beats_file : str, optional
        If provided, load beats from file. Will override beats_alg. Beats file must contain one beat
        timestamp per row in seconds.
    use_mfcc : bool, optional
        If True, use MFCC instead of fewshot features for shot-term similarity, by default False
    magicnorm : bool, optional
        Use normalization parameters required for deepsim model, by default True

    Returns
    -------
    np.ndarray
        Csync, deepsim beat-sync features
    np.ndarray 
        Msync, fewshot beat-sync features
    np.ndarray 
        Hsync, CQT (harmony) beat-sync features
    np.ndarray 
        beat_times, in seconds
    float
        audio_duration in seconds

    Raises
    ------
    Exception
        beats_algo must be "madmom" oro "librosa", otherwaise raises and error
    """
    # Use a temporary folder to resample audio 
    # Folder and all contents will get automatically deleted
    with tempfile.TemporaryDirectory() as tempdir:
        
        filename22 = os.path.join(tempdir, os.path.basename(os.path.splitext(filename)[0]) + "_22.wav")
        filename16 = os.path.join(tempdir, os.path.basename(os.path.splitext(filename)[0]) + "_16.wav")
        
        tfm = sox.Transformer()
        tfm.convert(samplerate=22050, n_channels=1)
        tfm.build(filename, filename22)

        tfm = sox.Transformer()
        tfm.convert(samplerate=16000, n_channels=1)
        tfm.build(filename, filename16)

        # Load audio @ 22kHz as y
        y, sr = librosa.load(filename22, sr=None)

        # compute file duration
        audio_duration = len(y)/sr

        # for super short files don't even try
        if audio_duration < 1:
            return None, None, None, None, audio_duration

        if beats_file is not None:
            # Load beat times in seconds and convert to frames
            # Add 0 and song-duration to beat times in seconds
            # return as "beats" (frames) and "beat_times" (seconds)
            beats, beat_times = load_beats(beats_file, audio_duration)
        else:
            if beats_alg=="madmom":
                beat_times = madmom_beats(filename)

                beats = librosa.time_to_frames(beat_times, sr=22050, hop_length=512)
                if beat_times[0] > 0:
                    beat_times = np.insert(beat_times, 0, 0)
                if beat_times[-1] < audio_duration:
                    beat_times = np.append(beat_times, audio_duration)
            
            elif beats_alg=="librosa":
                # Get tempo and beat locations in frames
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
                print_verbose(f"tempo: {tempo}", LOG)
                print_verbose(f"len(beats): {len(beats)}", LOG)

                # Get beat times in seconds
                maxframe = librosa.core.samples_to_frames(len(y)) + 1
                beat_times = librosa.frames_to_time(librosa.util.fix_frames(beats, x_min=0, x_max=maxframe), sr=sr)
            
            else:
                raise Exception("beats_alg must be either 'madmom' or 'librosa'")
        
        # Harmonic CQT
        # Perform HPSS to obtain harmonic wave signal yh
        yh = librosa.effects.harmonic(y, margin=8)
        C = librosa.amplitude_to_db(librosa.cqt(y=yh, sr=sr,
                                                bins_per_octave=BPO,
                                                n_bins=N_OCTAVES * BPO),
                                    ref=np.max)
        print_verbose("chroma {}".format(C.shape), LOG)
        # Synchronize (harmonic-derived) CQT to beats (framewise)
        Hsync = librosa.util.sync(C, beats, aggregate=np.median)
        
        # Compute deep similarity embeddings
        modelgrid = deepsim_model
        embedding_as_list = run_deepsim_inference(
            y, 
            modelgrid.base_model, 
            modelgrid.args, 
            modelgrid.session, 
            magicnorm=magicnorm).tolist()
        emb = np.asarray(embedding_as_list).T

        # Synchronize deepsim features to beats (framewise)
        Csync = librosa.util.sync(emb, beats)

        # Compute fewshot embeddings or MFCC
        if use_mfcc:
            # Compute MFCC from y
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            # Synchronize MFCC features to beats (framewise)
            Msync = librosa.util.sync(mfcc, beats)
        else:
            emb = run_fewshot_inference(filename16, fewshot_model).T
            # Synchronize fewshot features to beats (framewise)
            Msync = librosa.util.sync(emb, beats)

        # Return all embeddings/featires, beat times, and audio file duration in seconds
        return Csync, Msync, Hsync, beat_times, audio_duration


def segment_file(audiofile,
                 deepsim_model=None,
                 fewshot_model=None,
                 min_duration=8,
                 mu=0.5,
                 gamma=0.5,
                 beats_alg="madmom",
                 beats_file=None,
                 use_mfcc=False,
                 recurrence_width=9,
                 recurrence_smooth=9,
                 eigenvec_smooth=9):
    
    
    # Compute beat-synchronized features
    Csync, Msync, Hsync, beat_times, audio_duration = \
        make_beat_sync_features(audiofile,
                                deepsim_model, 
                                fewshot_model,
                                beats_alg=beats_alg,
                                beats_file=beats_file,
                                use_mfcc=use_mfcc,
                                magicnorm=MAGICNORM)

    features = {}
    features['Csync'] = Csync 
    features['Msync'] = Msync
    features['Hsync'] = Hsync
    features['beat_times'] = beat_times 
    features['audio_duration'] = audio_duration

    segmentations = segment_features(features, 
                                     min_duration=min_duration,
                                     mu=mu,
                                     gamma=gamma,
                                     recurrence_width=recurrence_width,
                                     recurrence_smooth=recurrence_smooth,                                           
                                     eigenvec_smooth=eigenvec_smooth)

    return segmentations, features
