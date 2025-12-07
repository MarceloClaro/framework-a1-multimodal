import os
from datetime import datetime
from typing import List, Dict, Any, Optional

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, DateTime, JSON, ForeignKey, Float
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

# URL de conexão do banco de dados (definida via variável de ambiente)
DB_URL = os.getenv(
    "FRAMEWORK_DB_URL",
    "postgresql+psycopg2://user:pass@localhost:5432/framework_a1",
)

engine = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class Run(Base):
    __tablename__ = "runs"
    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    domain = Column(String(50))
    task_type = Column(String(50))
    model_name = Column(String(100))

    classifications = relationship("Classification", back_populates="run")
    metrics = relationship("Metrics", back_populates="run", uselist=False)
    reports = relationship("Reports", back_populates="run", uselist=False)


class Classification(Base):
    __tablename__ = "classifications"
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("runs.id"))
    file_name = Column(String(255))
    file_hash = Column(String(64))
    predicted_label = Column(String(255))
    raw_model_output = Column(Text)
    run = relationship("Run", back_populates="classifications")


class Metrics(Base):
    __tablename__ = "metrics"
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("runs.id"))
    n_samples = Column(Integer)
    accuracy = Column(Float)
    macro_f1 = Column(Float, nullable=True)
    raw_json = Column(JSON)
    run = relationship("Run", back_populates="metrics")


class Reports(Base):
    __tablename__ = "reports"
    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("runs.id"))
    scientific = Column(Text)
    critic = Column(Text)
    replication = Column(Text)
    stats = Column(Text)
    evolutionary = Column(Text)
    specialized = Column(Text)
    run = relationship("Run", back_populates="reports")


def init_db():
    """Cria as tabelas no banco de dados se ainda não existirem."""
    Base.metadata.create_all(bind=engine)


def log_run(
    domain: str,
    task_type: str,
    model_name: str,
    results: List[Dict[str, Any]],
    metrics_dict: Optional[Dict[str, Any]],
    reports: Optional[Dict[str, str]],
) -> int:
    """Registra uma execução completa, retornando o ID do run salvo."""
    session = SessionLocal()
    try:
        run = Run(domain=domain, task_type=task_type, model_name=model_name)
        session.add(run)
        session.flush()  # Obtém run.id
        # Salva classificações
        for r in results:
            classification = Classification(
                run_id=run.id,
                file_name=r.get("file_name"),
                file_hash=r.get("hash"),
                predicted_label=r.get("predicted_label"),
                raw_model_output=r.get("raw_model_output"),
            )
            session.add(classification)
        # Salva métricas
        if metrics_dict is not None:
            metrics = Metrics(
                run_id=run.id,
                n_samples=metrics_dict.get("n_samples"),
                accuracy=metrics_dict.get("accuracy"),
                macro_f1=metrics_dict.get("macro_f1"),
                raw_json=metrics_dict,
            )
            session.add(metrics)
        # Salva relatórios
        if reports is not None:
            reports_row = Reports(
                run_id=run.id,
                scientific=reports.get("científico", ""),
                critic=reports.get("crítico", ""),
                replication=reports.get("replicação", ""),
                stats=reports.get("estatística", ""),
                evolutionary=reports.get("evolutivo", ""),
                specialized=reports.get("especializado", ""),
            )
            session.add(reports_row)
        session.commit()
        return run.id
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()
