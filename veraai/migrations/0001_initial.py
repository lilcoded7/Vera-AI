# Generated by Django 4.2.5 on 2023-09-28 03:38

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='VINSINCBUILDINGDATA',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('timestamp', models.DateTimeField()),
                ('mech_vent_nat_vent_infiltration', models.DecimalField(decimal_places=2, max_digits=10)),
                ('external_infiltration', models.DecimalField(decimal_places=2, max_digits=10)),
                ('task_lighting', models.DecimalField(decimal_places=2, max_digits=10)),
                ('general_lighting', models.DecimalField(decimal_places=2, max_digits=10)),
                ('miscellaneous', models.DecimalField(decimal_places=2, max_digits=10)),
                ('process', models.DecimalField(decimal_places=2, max_digits=10)),
                ('computer_equip', models.DecimalField(decimal_places=2, max_digits=10)),
                ('occupancy', models.DecimalField(decimal_places=2, max_digits=10)),
                ('solar_gains_exterior_windows', models.DecimalField(decimal_places=2, max_digits=10)),
                ('zone_sensible_cooling', models.DecimalField(decimal_places=2, max_digits=10)),
                ('air_temperature', models.DecimalField(decimal_places=2, max_digits=5)),
                ('radiant_temperature', models.DecimalField(decimal_places=2, max_digits=5)),
                ('operative_temperature', models.DecimalField(decimal_places=2, max_digits=5)),
                ('room_electricity', models.DecimalField(decimal_places=2, max_digits=10)),
                ('lighting', models.DecimalField(decimal_places=2, max_digits=10)),
                ('system_fans', models.DecimalField(decimal_places=2, max_digits=10)),
                ('system_pumps', models.DecimalField(decimal_places=2, max_digits=10)),
                ('cooling_electricity', models.DecimalField(decimal_places=2, max_digits=10)),
                ('exterior_lighting', models.DecimalField(decimal_places=2, max_digits=10)),
                ('outside_dry_bulb_temperature', models.DecimalField(decimal_places=2, max_digits=5)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Building',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('building_data', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='veraai.vinsincbuildingdata')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
