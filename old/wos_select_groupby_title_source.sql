USE wos;

SELECT title_source, title_source_abbrev, subject_extended, subject_traditional, heading, subheading
from papers
GROUP BY title_source, title_source_abbrev, subject_extended, subject_traditional, heading, subheading;
