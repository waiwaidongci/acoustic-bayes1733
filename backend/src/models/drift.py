from sqlalchemy import Column, Integer, String, Float, JSON, Text, ForeignKey, UniqueConstraint, Index
from ..config.database import Base


class ShiftPlan(Base):
    """一个来源下的归班版本。更正归班时另起 revision，旧版本保持冻结。"""
    __tablename__ = 'drift_shift_plans'
    __table_args__ = (UniqueConstraint('source_id', 'revision', name='uq_shift_plan_revision'),)
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey('acoustic_batches.id'), nullable=False, index=True)
    revision = Column(Integer, nullable=False)
    state = Column(String(20), default='draft', nullable=False)
    note = Column(Text, default='')
    created_by = Column(String(120), nullable=False)
    updated_by = Column(String(120), nullable=False)
    created_at = Column(String(40), nullable=False)


class ShiftAssignment(Base):
    """样本归班：哪个样本属于哪一班，以及班次的先后顺序。"""
    __tablename__ = 'drift_shift_assignments'
    __table_args__ = (
        UniqueConstraint('plan_id', 'reference', name='uq_shift_assignment_reference'),
        Index('ix_shift_assignment_order', 'plan_id', 'shift_order'),
    )
    id = Column(Integer, primary_key=True)
    plan_id = Column(Integer, ForeignKey('drift_shift_plans.id'), nullable=False)
    reference = Column(String(120), nullable=False)
    shift_name = Column(String(120), nullable=False)
    shift_order = Column(Integer, nullable=False)


class DriftJob(Base):
    """按指定故障计算频谱质心漂移分界的任务，可中断续跑。"""
    __tablename__ = 'drift_jobs'
    id = Column(Integer, primary_key=True)
    run_id = Column(Integer, ForeignKey('classifier_runs.id'), nullable=False, index=True)
    source_id = Column(Integer, ForeignKey('acoustic_batches.id'), nullable=False, index=True)
    plan_id = Column(Integer, ForeignKey('drift_shift_plans.id'), nullable=False)
    plan_revision = Column(Integer, nullable=False)
    label = Column(String(120), nullable=False)
    parameters = Column(JSON, nullable=False)
    state = Column(String(20), default='queued', nullable=False)
    checkpoint = Column(JSON, nullable=True)
    result = Column(JSON, nullable=True)
    error = Column(Text, default='')
    claim_token = Column(String(40), default='')
    created_at = Column(String(40), nullable=False)
    updated_at = Column(String(40), nullable=False)


class DriftSample(Base):
    """分批扫描得到的中转行，按 (任务, 样本编号) 唯一，重试不会重复计入。"""
    __tablename__ = 'drift_samples'
    __table_args__ = (
        UniqueConstraint('job_id', 'reference', name='uq_drift_sample_reference'),
        Index('ix_drift_sample_order', 'job_id', 'shift_order'),
    )
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey('drift_jobs.id'), nullable=False)
    reference = Column(String(120), nullable=False)
    shift_order = Column(Integer, nullable=False)
    shift_name = Column(String(120), nullable=False)
    centroid = Column(Float, nullable=False)


class DriftShift(Base):
    """每个班次已排序的频谱质心；分组完成后按班次幂等写入。"""
    __tablename__ = 'drift_shifts'
    __table_args__ = (UniqueConstraint('job_id', 'shift_order', name='uq_drift_shift_order'),)
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey('drift_jobs.id'), nullable=False)
    shift_order = Column(Integer, nullable=False)
    shift_name = Column(String(120), nullable=False)
    sample_count = Column(Integer, nullable=False)
    values = Column(JSON, nullable=False)


class DriftResult(Base):
    """每个来源+故障只保留一份已发布分界；发布用 revision 比较后替换。"""
    __tablename__ = 'drift_results'
    __table_args__ = (UniqueConstraint('source_id', 'label', name='uq_drift_result_source_label'),)
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey('acoustic_batches.id'), nullable=False)
    label = Column(String(120), nullable=False)
    run_id = Column(Integer, ForeignKey('classifier_runs.id'), nullable=False)
    plan_id = Column(Integer, nullable=False)
    plan_revision = Column(Integer, nullable=False)
    job_id = Column(Integer, nullable=False)
    result = Column(JSON, nullable=False)
    version = Column(Integer, default=1, nullable=False)
    published_by = Column(String(120), nullable=False)
    published_at = Column(String(40), nullable=False)


class DriftCost(Base):
    """动态规划已算班次的中转成本，续跑时从这里接上；重试幂等覆盖。"""
    __tablename__ = 'drift_costs'
    __table_args__ = (UniqueConstraint('job_id', 'shift_index', name='uq_drift_cost_shift'),)
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey('drift_jobs.id'), nullable=False)
    shift_index = Column(Integer, nullable=False)
    cost = Column(Float, nullable=False)
    full_deviation = Column(Float, nullable=True)
    full_median = Column(Float, nullable=True)
    full_count = Column(Integer, nullable=True)
