from django.db import migrations


def enable_pgvector(apps, schema_editor):
    if schema_editor.connection.vendor != 'postgresql':
        return

    with schema_editor.connection.cursor() as cursor:
        cursor.execute('CREATE EXTENSION IF NOT EXISTS vector;')


def disable_pgvector(apps, schema_editor):
    # Keep the extension in place on rollback so existing vector data remains readable.
    return


class Migration(migrations.Migration):

    dependencies = [
        ('attendance', '0004_notification'),
    ]

    operations = [
        migrations.RunPython(enable_pgvector, disable_pgvector),
    ]