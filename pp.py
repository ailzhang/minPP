from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Callable
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from functools import cmp_to_key


class WorkType(Enum):
    FWD = "fwd"
    BWD = "bwd"
    DGRAD = "dgrad"  # Dgrad is the gradient of activations
    WGRAD = "wgrad"  # Wgrad is the gradient of weights
    MIXED = "mixed"  # Overlapped fwd and bwd work items, e.g. in dualpipev case. this enum is for plotting purpose only.

    @property
    def is_fwd(self) -> bool:
        """Check if this work type is forward."""
        return self == WorkType.FWD

    @property
    def color(self) -> str:
        """Get the color for this work type."""
        if self == WorkType.FWD:
            return "#4F81BD"
        elif self == WorkType.BWD:
            return "#9BBB59"
        elif self == WorkType.WGRAD:
            return "#F79646"
        elif self == WorkType.DGRAD:
            return "#8064A2"
        elif self == WorkType.MIXED:
            return "#B94235"
        else:
            raise ValueError(f"Unknown work type: {self}")

    @property
    def duration(self) -> int:
        """Get the duration for this work type.
        Assumptions:
        1. Normally we have bwd = 2 * fwd by assuming GEMMs are dominant in the compute time.
        Note this is fwd and bwd time per layer for each microbatch with microbatch size of 1
        """
        if self == WorkType.FWD:
            return 1
        elif self == WorkType.BWD:
            return 2
        elif self == WorkType.WGRAD:
            return 1
        elif self == WorkType.DGRAD:
            return 1
        else:
            raise ValueError(f"Unknown work type: {self}")


@dataclass(frozen=True, kw_only=True)
class Microbatch:
    """Denote a microbatch."""

    idx: int


@dataclass(frozen=True, kw_only=True)
class WorkItem:
    mb: Microbatch
    start_layer_id: int
    label: WorkType = WorkType.FWD


@dataclass(frozen=True, kw_only=True)
class AssignedWork:
    """`work` item is started running `stage` from `start_timestamp`.

    Assumption:
    - Each FakePPrunner can only run one `AssignedWork` at a time.
    """

    start_timestamp: int
    # By default only one work item is assigned to eacher runner at a time.
    # In dualpipev case we can have 1 fwd and 1 bwd work item assigned to the same runner at the same time due to how we overlap the communication and compute in EP.
    work: list[WorkItem]

    def _get_fwd_bwd_items(self) -> tuple[WorkItem, WorkItem]:
        assert len(self.work) == 2, (
            f"Expected 2 work items, got {len(self.work)}: {self.work}"
        )
        fwd_item, bwd_item = self.work

        # We made sure fwd is first and bwd is second in the list.
        # This is just to be extra safe.
        if bwd_item.label == WorkType.FWD:
            bwd_item, fwd_item = fwd_item, bwd_item
        assert fwd_item.label.is_fwd and not bwd_item.label.is_fwd, (
            f"Expected one fwd and one bwd work item, got {fwd_item.label} and {bwd_item.label}"
        )
        return fwd_item, bwd_item

    @property
    def duration(self) -> int:
        """Get the duration of this assigned work."""
        if len(self.work) == 1:
            return self.work[0].label.duration
        else:
            fwd_item, bwd_item = self._get_fwd_bwd_items()
            return fwd_item.label.duration + bwd_item.label.duration
            # in dualpipev plot the duration is a sum.
            # return max(fwd_item.label.duration, bwd_item.label.duration)

    def textname(self, debug: bool = True) -> str:
        """Get the text name for this assigned work. Shows the microbatch index and layer id to run the microbatch."""
        if len(self.work) == 1:
            return (
                f"{self.work[0].mb.idx}-{self.work[0].start_layer_id}"
                if debug
                else f"{self.work[0].mb.idx}"
            )
        else:
            fwd_item, bwd_item = self._get_fwd_bwd_items()
            return f"f{fwd_item.mb.idx}|{fwd_item.start_layer_id}-b{bwd_item.mb.idx}|{bwd_item.start_layer_id}"


def plot_pp_schedule(
    schedule: dict[int, list[AssignedWork]],
    schedule_name: str,
    display_debug_info: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 3))
    yticks, ylabels = [], []

    num_devices = len(schedule)

    max_time = 0
    for device_id, segs in schedule.items():
        y_pos = 10 * (num_devices - 1 - device_id)  # invert vertical order
        yticks.append(y_pos + 5)
        ylabels.append(f"Device {device_id}")

        for seg in segs:
            duration = seg.duration
            color = (
                seg.work[0].label.color if len(seg.work) == 1 else WorkType.MIXED.color
            )
            textname = seg.textname(debug=display_debug_info)
            ax.broken_barh(
                [(seg.start_timestamp, duration)],
                (y_pos, 8),
                facecolors=color,
                edgecolors="black",
                linewidth=1,
            )
            # Add centered label in the segment
            ax.text(
                seg.start_timestamp + duration / 2,
                y_pos + 4,
                textname,  
                ha="center",
                va="center",
                fontsize=7,
            )
            max_time = max(max_time, seg.start_timestamp + duration)

    # Draw dotted vertical lines for every timestamp
    for t in range(max_time + 1):
        ax.axvline(x=t, color="gray", linestyle=":", linewidth=0.5)

    # Final layout adjustments
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylim(-2, 10 * num_devices)
    ax.set_xlim(0, max_time)
    ax.set_xticks(range(0, max_time + 1))
    ax.set_xlabel("Time")
    ax.grid(False)

    # Legend
    ax.legend(
        handles=[Patch(color=label.color, label=label.value) for label in WorkType],
        loc="upper left",
        bbox_to_anchor=(1.02, 1),  # (x_offset, y_offset) relative to axes
        borderaxespad=0,
    )
    ax.set_title(f"{schedule_name}")

    plt.tight_layout()
    # Save the plot to a file
    plt.savefig(f"{schedule_name}.png", dpi=300, bbox_inches="tight")


class ScheduleAlgo(Enum):
    """Algorithm to use during execution"""

    DFS = "dfs"
    BFS = "bfs"


@dataclass(frozen=True, kw_only=True)
class FullModelList:
    """Number of layers in the whole model.

    Assumptions:
      1. All layers are uniform to make partitioning easy.
    """

    num_layers: int


@dataclass(frozen=True, kw_only=True)
class PPStage:
    """Paritioned from the whole model, consecutive layers from [start_layer_id, end_layer_id] (inclusive)"""

    start_layer_id: int
    end_layer_id: int
    is_last_stage: bool = False

    @property
    def is_first_stage(self) -> bool:
        return self.start_layer_id == 0

    def __str__(self) -> str:
        return f"PPStage([{self.start_layer_id}, {self.end_layer_id}], is_last={self.is_last_stage}])"


def compare_work_dfs(x: WorkItem, y: WorkItem) -> int:
    """Compare two backward work items based on their microbatch index and label.
    - zero bubble: dgrad > fwd > wgrad
    - bwd > fwd
    """
    if x.label.is_fwd and y.label.is_fwd:
        # If both are forward work items, larger start_layer_id go first
        if x.start_layer_id > y.start_layer_id:
            return -1
        elif x.start_layer_id < y.start_layer_id:
            return 1
        return x.mb.idx - y.mb.idx
    if not x.label.is_fwd and not y.label.is_fwd:
        # If both are backward work items, compare by label first, dgrad always comes before wgrad
        if x.label == WorkType.DGRAD and y.label == WorkType.WGRAD:
            return -1
        if x.label == WorkType.WGRAD and y.label == WorkType.DGRAD:
            return 1
        # If both are same type of backward work items, compare by microbatch index
        return x.mb.idx - y.mb.idx
    # One item is forward and the other is backward, prioritize dgrad first, then fwd, then wgrad
    if x.label.is_fwd and not y.label.is_fwd:
        # x is fwd, y is bwd
        if y.label == WorkType.DGRAD:
            return 1
        elif y.label == WorkType.WGRAD:
            return -1
        else:
            return 1
    if not x.label.is_fwd and y.label.is_fwd:
        # x is bwd, y is fwd
        if x.label == WorkType.DGRAD:
            return -1
        elif x.label == WorkType.WGRAD:
            return 1
        else:
            return -1
    return 0  # This should not happen, but just in case


def compare_work_bfs(x: WorkItem, y: WorkItem) -> int:
    """Compare two work items based on their start_layer_id and microbatch index.
    - forward work items are prioritized over backward work items
    - within the same type, compare by start_layer_id
    - zero bubble : dgrad > wgrad
    """
    if x.label.is_fwd and y.label.is_fwd:
        # If both are forward work items, smaller start_layer_id go first
        if x.start_layer_id == y.start_layer_id:
            return x.mb.idx - y.mb.idx
        return x.start_layer_id - y.start_layer_id
    if not x.label.is_fwd and not y.label.is_fwd:
        # If both are backward work items, larger start_layer_id go first
        if x.start_layer_id == y.start_layer_id:
            return x.mb.idx - y.mb.idx
        return y.start_layer_id - x.start_layer_id
    # One item is forward and the other is backward, prioritize forward first
    if x.label.is_fwd and not y.label.is_fwd:
        # x is fwd, y is bwd
        return -1  # forward goes first
    if not x.label.is_fwd and y.label.is_fwd:
        # x is bwd, y is fwd
        return 1  # forward goes first
    return 0  # This should not happen, but just in case


@dataclass(frozen=False, kw_only=True)
class FakePPrunner:
    """Local shard of the whole Model on each device.

    Note this dataclass is not frozen as we need to carry some state
    """

    rank: int
    num_runners: int  # Number of total runners in the pipeline
    local_model_chunk: list[PPStage]
    current_item: AssignedWork | None = None
    # This is used to record the segments assigned to each worker for plotting later.
    assigned_segments: list[AssignedWork] = field(default_factory=list)
    inflight_batches: int = 0
    zero_bubble: bool = False  # Whether to use zero bubble optimization

    def busy_until(self) -> int:
        assert self.current_item is not None
        return self.current_item.start_timestamp + self.current_item.duration

    def is_idle(self, current_timestamp: int) -> bool:
        if self.current_item is None:
            return True
        return current_timestamp > self.busy_until()

    def get_stage_idx(self, start_layer_id: int) -> int:
        """Get the index of the stage that contains the start_layer_id."""
        for idx, stage in enumerate(self.local_model_chunk):
            if (
                stage.start_layer_id == start_layer_id
                or stage.end_layer_id == start_layer_id
            ):
                return idx
        raise ValueError(
            f"start_layer_id {start_layer_id} not found in local_model_chunk: {self.local_model_chunk}"
        )

    def is_work_item_for_stage(self, work_item: WorkItem) -> bool:
        """Check if the work_item is for any of the stages in this runner."""
        for stage in self.local_model_chunk:
            if (
                stage.start_layer_id == work_item.start_layer_id
                or stage.end_layer_id == work_item.start_layer_id
            ):
                return True
        return False

    def _assign_work_to_rank(
        self, start_timestamp: int, work_item: list[WorkItem]
    ) -> None:
        """Assign work_item to worker with rank at start_timestamp."""
        for item in work_item:
            stage_idx = self.get_stage_idx(start_layer_id=item.start_layer_id)
            print(
                f" worker {self.rank} started mb {item.mb.idx} {item.label} for stage {self.local_model_chunk[stage_idx]} at timestamp {start_timestamp}"
            )
            if item.label.is_fwd:
                # If this is a forward work item, we increment inflight_batches
                self.inflight_batches += 1
            elif item.label == WorkType.BWD or item.label == WorkType.DGRAD:
                # If this is a backward work item, we decrement inflight_batches
                # DGRAD is always handled first if exists, so we only decrement inflight_batches once.
                # TODO: maybe make inflight_batches a set to record the index
                self.inflight_batches -= 1

        self.current_item = AssignedWork(
            start_timestamp=start_timestamp,
            work=work_item,
        )
        # Record the segment for plotting later
        self.assigned_segments.append(self.current_item)

    def assign_work_bfs(
        self, start_timestamp: int, input_queue: list[WorkItem]
    ) -> None:
        """Assign work from input_queue to worker with rank at start_timestamp using BFS."""
        work_items_for_this_rank = [
            item for item in input_queue if self.is_work_item_for_stage(item)
        ]
        highest_pri_work = (
            sorted(work_items_for_this_rank, key=cmp_to_key(compare_work_bfs))[0]
            if work_items_for_this_rank
            else None
        )
        if highest_pri_work is not None:
            self._assign_work_to_rank(
                start_timestamp=start_timestamp,
                work_item=[highest_pri_work],
            )
            input_queue.remove(highest_pri_work)
            return

    def assign_work_dfs(
        self,
        start_timestamp: int,
        input_queue: list[WorkItem],
        max_inflight_batches: int,
        max_assigned_items: int = 1,
    ) -> None:
        """Assign work from input_queue to worker with rank at start_timestamp using DFS.
        Normally we assign the highest priority work item from the input queue to the worker, in dualpipev case we assign at most 1 fwd and 1 bwd work items to the same worker."""
        work_items_for_current_rank = [
            item for item in input_queue if self.is_work_item_for_stage(item)
        ]
        if work_items_for_current_rank:
            sorted_work_items = sorted(
                work_items_for_current_rank, key=cmp_to_key(compare_work_dfs)
            )
            # highest_pri_work is first item in the sorted list, and self.inflight_batches is smaller than max_inflight_batches
            if max_assigned_items == 1:
                highest_pri_work = None
                for work_item in sorted_work_items:
                    if (
                        not work_item.label.is_fwd
                        or self.inflight_batches < max_inflight_batches
                    ):
                        highest_pri_work = work_item
                        break
                if highest_pri_work is not None:
                    self._assign_work_to_rank(
                        start_timestamp=start_timestamp,
                        work_item=[highest_pri_work],
                    )
                    input_queue.remove(highest_pri_work)
            else:
                # Dualpipev case, we can assign 1 fwd and 1 bwd work items to the same worker
                # Note since we already broke the backward item into WGRAD and DGRAD, we need to get the highest priority forward
                # dgrad+wgrad.
                # TODO: simplify this logic, it is a bit convoluted.
                highest_fwd_work = None
                highest_bwd_work = None
                for work_item in sorted_work_items:
                    if (
                        work_item.label.is_fwd
                        and self.inflight_batches <= max_inflight_batches
                    ):
                        # note here self.inflight_batches can be equal to max_inflight_batches, since when we run a fwd and bwd at the same time, we don't increase the total number of inflight microbatches.
                        if highest_fwd_work is None:
                            highest_fwd_work = work_item
                    elif not work_item.label.is_fwd:
                        if highest_bwd_work is None:
                            highest_bwd_work = work_item

                if (
                    highest_fwd_work is not None
                    and highest_bwd_work is not None
                    and highest_bwd_work.label == WorkType.DGRAD
                    and highest_fwd_work.mb.idx - highest_bwd_work.mb.idx
                    == self.num_runners
                ):
                    # the wgrad for highest_bwd_work must be still in the input queue
                    whole_backward_work = WorkItem(
                        mb=highest_bwd_work.mb,
                        start_layer_id=highest_bwd_work.start_layer_id,
                        label=WorkType.BWD,
                    )
                    self._assign_work_to_rank(
                        start_timestamp=start_timestamp,
                        work_item=[highest_fwd_work, whole_backward_work],
                    )
                    wgrad_to_remove = WorkItem(
                        mb=highest_bwd_work.mb,
                        start_layer_id=highest_bwd_work.start_layer_id,
                        label=WorkType.WGRAD,
                    )
                    input_queue.remove(highest_fwd_work)
                    input_queue.remove(highest_bwd_work)
                    input_queue.remove(wgrad_to_remove)
                elif highest_bwd_work is not None:
                    self._assign_work_to_rank(
                        start_timestamp=start_timestamp,
                        work_item=[highest_bwd_work],
                    )
                    input_queue.remove(highest_bwd_work)
                elif (
                    highest_fwd_work is not None
                    and self.inflight_batches < max_inflight_batches
                ):
                    self._assign_work_to_rank(
                        start_timestamp=start_timestamp,
                        work_item=[highest_fwd_work],
                    )
                    input_queue.remove(highest_fwd_work)
                else:
                    # If no work items were found, we can stop the search
                    pass

    def _get_backward_work_item(
        self, mb: Microbatch, start_layer_id: int
    ) -> list[WorkItem]:
        if self.zero_bubble:
            # If zero bubble optimization is enabled, we return both WGRAD and DGRAD work items
            return [
                WorkItem(mb=mb, start_layer_id=start_layer_id, label=WorkType.WGRAD),
                WorkItem(mb=mb, start_layer_id=start_layer_id, label=WorkType.DGRAD),
            ]
        else:
            # Otherwise, we only return BWD work item
            return [WorkItem(mb=mb, start_layer_id=start_layer_id, label=WorkType.BWD)]

    def get_next_layer_id_and_work_item(
        self, finished_item: AssignedWork
    ) -> list[WorkItem]:
        """Input for stage start_layer_id is for fwd, input for stage end_layer_id is for bwd.

        Note this knowledge is local to FakePPrunner class.
        """
        return_items = []
        for item in finished_item.work:
            local_stage_idx = self.get_stage_idx(start_layer_id=item.start_layer_id)
            stage = self.local_model_chunk[local_stage_idx]
            if item.label.is_fwd:
                if stage.is_last_stage:
                    # next work item is still for the same stage, but bwd
                    return_items.extend(
                        self._get_backward_work_item(item.mb, stage.end_layer_id)
                    )
                else:
                    # next work item is for the next stage, still fwd
                    return_items.append(
                        WorkItem(
                            mb=item.mb,
                            start_layer_id=stage.end_layer_id + 1,
                            label=WorkType.FWD,
                        )
                    )
            else:
                if stage.is_first_stage:
                    # If it's the first stage, all backward work is done
                    pass
                elif item.label == WorkType.DGRAD or item.label == WorkType.BWD:
                    # backward for the previous stage is unblocked, wgrad and dgrad
                    return_items.extend(
                        self._get_backward_work_item(item.mb, stage.start_layer_id - 1)
                    )
                else:
                    # Nothing need to be done for DGRAD, as it is handled locally and done.
                    pass
        return return_items

    def run_for_one_sec(self, current_timestamp: int) -> list[WorkItem]:
        """Run for one second, if work is done and can be sent to the next stage, return (next_layer_id, work_item).

        Assumption: current_item for this second is already assigned by the PPScheduler or it stays idle for the second.
        """
        work_item = []
        if self.current_item is None:
            # Nothing is running, idle for 1 second
            return work_item
        if self.busy_until() == current_timestamp + 1:
            print(
                f" worker {self.rank} finished {self.current_item.textname()} at timestamp {current_timestamp + self.current_item.duration}"
            )
            # This item is done.
            # - mark this worker as free
            # - and pass it on to next stage.
            # Gpipe only has one stage per worker
            work_item = self.get_next_layer_id_and_work_item(self.current_item)
            self.current_item = None
        else:
            assert current_timestamp + 1 < self.busy_until(), (
                f"working item must be processed before {current_timestamp}"
            )
        return work_item


def create_all_stages_evenly(
    full_model: FullModelList, num_stages: int
) -> list[PPStage]:
    # Don't worry about edge case now
    assert full_model.num_layers % num_stages == 0, (
        "num_layers in full_model must be divisble by num_stages"
    )
    num_layers_per_stage = full_model.num_layers // num_stages
    return [
        PPStage(
            start_layer_id=i * num_layers_per_stage,
            end_layer_id=(i + 1) * num_layers_per_stage - 1,
            is_last_stage=(i == num_stages - 1),
        )
        for i in range(num_stages)
    ]


def create_one_stage_per_runner(
    all_stages: list[PPStage], zero_bubble: bool = False
) -> list[FakePPrunner]:
    return [
        FakePPrunner(
            rank=i,
            num_runners=len(all_stages),
            local_model_chunk=[stage],
            zero_bubble=zero_bubble,
        )
        for i, stage in enumerate(all_stages)
    ]


def create_runners_with_looping_assignment(
    all_stages: list[PPStage], pp_size: int
) -> list[FakePPrunner]:
    """Create runners with looping assignment, so that each runner has a chunk of stages."""
    num_stages = len(all_stages)
    assert num_stages % pp_size == 0, "num_stages must be divisble by pp_size"
    return [
        FakePPrunner(
            rank=i,
            num_runners=pp_size,
            local_model_chunk=[all_stages[j] for j in range(i, num_stages, pp_size)],
            # Each runner gets a chunk of stages, looping through all stages
            # e.g. pp_size=4, num_stages=8, then each runner gets 2 stages:
            # rank 0: stages 0, 4
            # rank 1: stages 1, 5
            # rank 2: stages 2, 6
            # rank 3: stages 3, 7
        )
        for i in range(pp_size)
    ]


def create_runners_with_vshape_zb_assignment(
    all_stages: list[PPStage], pp_size: int
) -> list[FakePPrunner]:
    """Create runners with V-shape assignment, so that each runner has a chunk of stages."""
    num_stages = len(all_stages)
    assert num_stages % pp_size == 0, "num_stages must be divisble by pp_size"
    # V-shape assignment means that each runner gets a chunk of stages, but the chunks are folded in a v shape mapping of devices.
    # e.g. pp_size=4, num_stages=8, then each runner gets 2 stages:
    # rank 0: stages 0, 7
    # rank 1: stages 1, 6
    # rank 2: stages 2, 5
    # rank 3: stages 3, 4
    num_stages_per_runner = num_stages // pp_size
    assert num_stages_per_runner % 2 == 0, (
        "num_stages_per_runner must be even for v-shape assignment"
    )
    runners = []
    for i in range(pp_size):
        local_stages = []
        for j in range(num_stages_per_runner):
            if j % 2 == 0:
                # Even index, take from the start of the list
                stage_idx = (i + j // 2 * pp_size) % num_stages
            else:
                # Odd index, take from the end of the list
                stage_idx = (num_stages - 1 - i - j // 2 * pp_size) % num_stages
            local_stages.append(all_stages[stage_idx])
        print(f"rank {i} has stages: {[str(stage) for stage in local_stages]}")
        runners.append(
            FakePPrunner(
                rank=i,
                num_runners=pp_size,
                local_model_chunk=local_stages,
                zero_bubble=True,
            )
        )
    return runners


@dataclass(frozen=True, kw_only=True)
class PPSchedule:
    workers: list[FakePPrunner]
    algo: ScheduleAlgo
    type: str
    max_inflight_batch_fn: Callable | None = None
    delayed_work_items: list[WorkItem] = field(default_factory=list) 

    @classmethod
    def create_gpipe(cls, full_model: FullModelList, pp_size: int):
        # Gpipe splits the full_model evenly to pp_size stages, one stage per device
        all_stages = create_all_stages_evenly(full_model=full_model, num_stages=pp_size)
        workers = create_one_stage_per_runner(all_stages)
        return cls(workers=workers, algo=ScheduleAlgo.BFS, type="gpipe")

    @classmethod
    def create_1f1b(cls, full_model: FullModelList, pp_size: int):
        # 1F1B splits the full_model evenly to pp_size stages, one stage per device
        all_stages = create_all_stages_evenly(full_model=full_model, num_stages=pp_size)
        workers = create_one_stage_per_runner(all_stages)
        return cls(
            workers=workers,
            algo=ScheduleAlgo.DFS,
            max_inflight_batch_fn=lambda rank: pp_size - rank,
            type="1f1b",
        )

    @classmethod
    def create_zb1f1b(cls, full_model: FullModelList, pp_size: int):
        # Zero bubble 1F1B splits the full_model evenly to pp_size stages, one stage per device
        # The difference between 1F1B and ZB1F1B is that ZB1F1B splits the backward to wgrad and dgrad and it priortize dgrad first.
        all_stages = create_all_stages_evenly(full_model=full_model, num_stages=pp_size)
        workers = create_one_stage_per_runner(all_stages, zero_bubble=True)
        return cls(
            workers=workers,
            algo=ScheduleAlgo.DFS,
            max_inflight_batch_fn=lambda rank: pp_size - rank,
            type="zb1f1b",
        )

    @classmethod
    def create_eager1f1b(cls, full_model: FullModelList, pp_size: int):
        # Eager 1F1B splits the full_model evenly to pp_size stages, one stage per device
        all_stages = create_all_stages_evenly(full_model=full_model, num_stages=pp_size)
        workers = create_one_stage_per_runner(all_stages)
        return cls(
            workers=workers,
            algo=ScheduleAlgo.DFS,
            max_inflight_batch_fn=lambda rank: 2 * (pp_size - rank) - 1,
            type="eager1f1b",
        )

    @classmethod
    def create_vshape_zb(
        cls, full_model: FullModelList, pp_size: int, layers_per_stage: int
    ):
        """Create a virtual pipeline with V-shape assignment."""
        assert full_model.num_layers % layers_per_stage == 0, (
            "num_layers in full_model must be divisble by layers_per_stage"
        )
        num_stages = full_model.num_layers // layers_per_stage
        all_stages = create_all_stages_evenly(
            full_model=full_model, num_stages=num_stages
        )

        workers = create_runners_with_vshape_zb_assignment(
            all_stages=all_stages,
            pp_size=pp_size,
        )
        return cls(
            workers=workers,
            algo=ScheduleAlgo.DFS,
            type="vshape_zb",
            max_inflight_batch_fn=lambda rank: 2 * pp_size,
        )

    @classmethod
    def create_dualpipe_vshape(
        cls, full_model: FullModelList, pp_size: int, layers_per_stage: int
    ):
        """Create a dualpipe vshape schedule. The original dualpipe schedule is bad for its 2x memory usage. dualpipe vshape is a better version.'
        see https://github.com/deepseek-ai/DualPipe#dualpipev and https://hackmd.io/@ufotalent/r1lVXsa9Jg#Cut-in-half-is-an-EP-specialized-ZB-V-Schedule
        dualpipev is just a variant of vshape_zb, but with an optimization based on sparse MoE.
        """
        assert full_model.num_layers % layers_per_stage == 0, (
            "num_layers in full_model must be divisble by layers_per_stage"
        )
        num_stages = full_model.num_layers // layers_per_stage
        all_stages = create_all_stages_evenly(
            full_model=full_model, num_stages=num_stages
        )
        workers = create_runners_with_vshape_zb_assignment(
            all_stages=all_stages, pp_size=pp_size
        )
        return cls(
            workers=workers,
            algo=ScheduleAlgo.DFS,
            max_inflight_batch_fn=lambda rank: 2 * pp_size,
            type="dualpipev",
        )

    @classmethod
    def create_interleaved_virtual_pipeline(
        cls, full_model: FullModelList, pp_size: int, layers_per_stage: int
    ):
        """Create a virtual pipeline with layers_per_stage layers per stage.

        This is Megatron style interleaved pipeline parallelism, where each stage has multiple layers.
        """
        assert full_model.num_layers % layers_per_stage == 0, (
            "num_layers in full_model must be divisble by layers_per_stage"
        )
        num_stages = full_model.num_layers // layers_per_stage
        all_stages = create_all_stages_evenly(
            full_model=full_model, num_stages=num_stages
        )

        workers = create_runners_with_looping_assignment(
            all_stages=all_stages, pp_size=pp_size
        )

        return cls(
            workers=workers,
            algo=ScheduleAlgo.DFS,
            max_inflight_batch_fn=lambda rank: 11
            - 2 * rank,  # FIXME: hardcoded for 4 ranks, should be a function of pp_size
            type="interleaved",
        )

    @classmethod
    def create_looped_bfs(
        cls, full_model: FullModelList, pp_size: int, layers_per_stage: int
    ):
        """Create a virtual pipeline with layers_per_stage layers per stage.

        This is useful for simulating a pipeline with more than one layer per stage.
        """
        assert full_model.num_layers % layers_per_stage == 0, (
            "num_layers in full_model must be divisble by layers_per_stage"
        )
        num_stages = full_model.num_layers // layers_per_stage
        all_stages = create_all_stages_evenly(
            full_model=full_model, num_stages=num_stages
        )

        assert num_stages % pp_size == 0, "num_stages must be divisble by pp_size"
        workers = create_runners_with_looping_assignment(
            all_stages=all_stages, pp_size=pp_size
        )
        return cls(workers=workers, algo=ScheduleAlgo.BFS, type="looped_bfs")

    def assign_work_from_input_queue(
        self, start_timestamp: int, input_queue: list[WorkItem]
    ) -> None:
        """Only consume from input_queue and assign to workers, no addition to input_queue"""
        # Note: in practice, this for-loop is run in parallel by multiple workers.
        for worker in self.workers:
            if worker.current_item:
                # Still in the middle of something, skip fetching
                continue
            match self.algo:
                case ScheduleAlgo.BFS:
                    # BFS scheduling
                    worker.assign_work_bfs(
                        start_timestamp=start_timestamp, input_queue=input_queue
                    )
                case ScheduleAlgo.DFS:
                    # DFS scheduling
                    assert self.max_inflight_batch_fn is not None, (
                        "max_inflight_batch_fn must be set for DFS scheduling"
                    )
                    worker.assign_work_dfs(
                        start_timestamp=start_timestamp,
                        input_queue=input_queue,
                        max_inflight_batches=self.max_inflight_batch_fn(worker.rank),
                        max_assigned_items=1 if self.type != "dualpipev" else 2,
                    )
                case _:
                    raise ValueError(f"Unknown scheduling algorithm: {self.algo}")

    def run_for_one_sec(self, current_timestamp: int, input_queue) -> None:
        # At current_timestamp, assign work from input_queue to start on worker, record the assignment for plotting later.
        self.assign_work_from_input_queue(
            start_timestamp=current_timestamp, input_queue=input_queue
        )

        # Let all worker finish running for [current_timestamp, current_timestamp+1), clear current_item if it's done, append new work items to the input_queue
        # Here we assume transfer between stages is instant, so once a work item is done, its next item will be immediately available for the next stage.
        # Some schedules take the data transfer time into account and try to overlap the transfer with computation,
        # e.g. eager1f1b does this but it doesn't explicitly model the transfer time.
        # interleaved virtual pipeline does this by delaying bwd work items by 1 time unit. 
        # This is not very consistent in their plot, but it is a valid assumption.

        # Note: in practice, this for-loop is run in parallel by multiple workers.
        delayed_work_items: list[WorkItem] = []
        for worker in self.workers:
            work_item = worker.run_for_one_sec(current_timestamp)
            if self.type == "interleaved":
                for item in work_item:
                    if item.label == WorkType.BWD and not worker.is_work_item_for_stage(item):
                        # if backward is not on the same runner, we delay it by 1 second
                        delayed_work_items.append(item)
                        work_item.remove(item)
            if work_item:
                input_queue.extend(work_item)
        if self.type == "interleaved": 
            input_queue.extend(self.delayed_work_items)
            self.delayed_work_items.clear()  # Clear the delayed work items after processing
            self.delayed_work_items.extend(delayed_work_items)  # Clear the delayed work items after processing

    def run_schedule_and_plot(self, num_microbatches: int) -> None:
        input_queue: list[WorkItem] = []

        # At timestamp 0, all microbatches are ready for layer 0.
        timestamp = 0
        # mb_idx is 1-indexed in many papers
        # In practice, input_queue is not global, but per rank, and ranks communicate via p2p send/recv.
        input_queue = [
            WorkItem(mb=Microbatch(idx=i + 1), start_layer_id=0, label=WorkType.FWD)
            for i in range(num_microbatches)
        ]

        while True:
            print(f"==== [{timestamp}, {timestamp + 1})...")
            # All work is done if
            #  1. no input work items for any stages
            #  2. and all workers are idle.
            num_remaining_work_items = len(input_queue) + len(self.delayed_work_items)
            if num_remaining_work_items == 0:
                is_worker_idle = [
                    worker.is_idle(current_timestamp=timestamp)
                    for worker in self.workers
                ]
                if all(is_worker_idle):
                    break

            # We assume data transfer between stages is instant, so we first need to scan every worker to process finished work items and do "fake transfer" first.
            self.run_for_one_sec(timestamp, input_queue)
            timestamp += 1

        # Done running the schedule, now plot the schedule.
        self.plot()

    def plot(self) -> None:
        """Plot the schedule of all workers."""
        print(
            f"Plotting schedule for {self.type} schedule with {len(self.workers)} workers..."
        )
        plot_pp_schedule(
            {
                rank: worker.assigned_segments
                for rank, worker in enumerate(self.workers)
            },
            schedule_name=f"{self.type}",
            display_debug_info=len(self.workers[0].local_model_chunk)
            > 1,  # if more than one stage, display debug info to make the plot more informative
        )


def main() -> None:
    num_devices = 4
    full_model = FullModelList(num_layers=8)
    schedules = [
        PPSchedule.create_gpipe(full_model=full_model, pp_size=num_devices),
        PPSchedule.create_1f1b(full_model=full_model, pp_size=num_devices),
        PPSchedule.create_zb1f1b(full_model=full_model, pp_size=num_devices),
        PPSchedule.create_eager1f1b(full_model=full_model, pp_size=num_devices),
        PPSchedule.create_looped_bfs(
            full_model=full_model, pp_size=num_devices, layers_per_stage=1
        ),
        PPSchedule.create_interleaved_virtual_pipeline(
            full_model=full_model, pp_size=num_devices, layers_per_stage=1
        ),
        PPSchedule.create_vshape_zb(
            full_model=full_model, pp_size=num_devices, layers_per_stage=1
        ),  # although look slightly different from the one in the paper, both are valid
    ]

    for schedule in schedules:
        schedule.run_schedule_and_plot(num_microbatches=8)

    # Dualpipev schedule runs 10 microbatches
    schedule = PPSchedule.create_dualpipe_vshape(
        full_model=full_model, pp_size=num_devices, layers_per_stage=1
    )  # although look slightly different from the one in the paper, both are valid
    schedule.run_schedule_and_plot(num_microbatches=10)


if __name__ == "__main__":
    main()
