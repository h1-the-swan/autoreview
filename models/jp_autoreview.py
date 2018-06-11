from datetime import datetime

from sqlalchemy import Column, Integer, BigInteger, String, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class PipelineTest(Base):
    __tablename__ = 'pipeline_tests'
    id = Column(Integer, primary_key=True)
    datetime_added = Column(DateTime, default=datetime.now())
    review_paper_id = Column(BigInteger)
    random_seed = Column(Integer)
    datadir = Column(Text)
    num_correctly_predicted = Column(Integer)
    num_target_papers = Column(Integer)
    num_target_in_candidates = Column(Integer)
    num_seed_papers = Column(Integer)
    score_correctly_predicted = Column(Float)
    num_candidates = Column(Integer)
    features = Column(Text)
    clf = Column(Text)
    clf_type = Column(String(64))
    time_fit = Column(Integer)
    time_predict = Column(Integer)
    saved_model = Column(Text)
