<?php

namespace Database\Seeders;

use App\Models\DataKtp;
use Illuminate\Database\Console\Seeds\WithoutModelEvents;
use Illuminate\Database\Seeder;

class DataKtpSeeder extends Seeder
{
    /**
     * Run the database seeds.
     */
    public function run(): void
    {
        $data = [
            [
                'nama' => 'Budi Santoso',
                'nik' => '1234567890123456',
                'alamat' => 'Jl. Melati No. 12, Jakarta',
                'ttl' => 'Jakarta, 1 Januari 1990',
                'jk' => 'Laki-laki',
                'agama' => 'Islam',
            ],
            [
                'nama' => 'Siti Aminah',
                'nik' => '2345678901234567',
                'alamat' => 'Jl. Mawar No. 45, Surabaya',
                'ttl' => 'Surabaya, 12 Februari 1992',
                'jk' => 'Perempuan',
                'agama' => 'Islam',
            ],
            [
                'nama' => 'Andi Wijaya',
                'nik' => '3456789012345678',
                'alamat' => 'Jl. Kenanga No. 7, Bandung',
                'ttl' => 'Bandung, 23 Maret 1988',
                'jk' => 'Laki-laki',
                'agama' => 'Kristen',
            ],
        ];

        foreach ($data as $item) {
            DataKtp::create($item);
        }
    }
}
