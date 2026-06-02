# Generated manually for attendance decision evidence fields.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('attendance', '0006_faceembedding_quality_score'),
    ]

    operations = [
        migrations.AlterField(
            model_name='attendancelog',
            name='status',
            field=models.CharField(
                choices=[
                    ('ON_TIME', 'Đúng giờ'),
                    ('LATE', 'Đi muộn'),
                    ('ABSENT', 'Vắng mặt'),
                    ('MANUAL_REVIEW', 'Cần duyệt'),
                    ('SUSPICIOUS', 'Nghi ngờ gian lận'),
                ],
                default='ON_TIME',
                max_length=20,
            ),
        ),
        migrations.AddField(
            model_name='attendancelog',
            name='face_confidence',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='attendancelog',
            name='liveness_score',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='attendancelog',
            name='spoof_score',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='attendancelog',
            name='replay_score',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='attendancelog',
            name='risk_score',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='attendancelog',
            name='recognition_method',
            field=models.CharField(blank=True, default='', max_length=50),
        ),
        migrations.AddField(
            model_name='attendancelog',
            name='top_candidate_score',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='attendancelog',
            name='second_candidate_score',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='attendancelog',
            name='decision_reason',
            field=models.TextField(blank=True, default=''),
        ),
        migrations.AddField(
            model_name='attendancelog',
            name='evidence',
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
